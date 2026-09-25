# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Phạm Ngọc Đông

Công cụ gán nhãn đã dùng: CVAT v2.76.0 chạy bằng Docker trên máy cá nhân; import/export định dạng
Ultralytics YOLO Detection 1.0.

Nguồn số liệu: `reports/rounds_table.md`, `outputs/metrics_round0.json`, `outputs/metrics_round1.json`,
`outputs/selection_round1.csv`, `outputs/round1_diff.md`, `outputs/compare_round0.jpg`,
`outputs/compare_round1.jpg`. Nhãn test do mô hình tạo, chưa được người rà, nên mọi số đo dưới đây là
**mức khớp với bộ tham chiếu**, không phải độ chính xác tuyệt đối.

## 1. Dữ liệu và cách chia tập

Video quay bằng camera cố định, lấy 2.5 frame/giây, nên hai frame liền nhau (cách 0.4 s) gần như giống hệt,
và mỗi xe ở trong khung hình vài giây. Nếu chia ngẫu nhiên, cùng một chiếc xe ở cùng vị trí sẽ xuất hiện
cả trong ảnh train lẫn ảnh test. Khi đó model được chấm trên những xe nó đã học, nên số đo trên test bị
**lệch lên (lạc quan hơn thực tế)**. Đây là rò rỉ dữ liệu.

Bộ dữ liệu chia theo thời gian: 20 ảnh test ở 4 đoạn có tâm 20/60/100/140 s, 112 ảnh vùng đệm bị loại
(4 s trước và sau mỗi đoạn test), 268 ảnh còn lại làm pool. Ảnh pool gần test nhất vẫn cách 4.4 s
(`data/DATA.md`). Tôi không sửa `data/test/labels/` và không đưa ảnh test vào lô gán nhãn;
`check_submission.py` xác nhận hash nhãn test khớp bản phát hành.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 trong `rounds_table.md` (20 ảnh test, 403 box tham chiếu, bỏ qua 14 box cao dưới 16 px):

| vòng | model | AP50 | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0.771 | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Tại conf 0.25, model cold start có TP 197, FP 16, FN 206 (`metrics_round0.json`). Model khá chính xác khi
đã báo (precision 0.925) nhưng **bỏ sót khoảng một nửa số xe** (recall 0.489). Recall theo kích thước:
xe nhỏ chỉ 0.182 (66 box), xe trung bình 0.547, xe lớn 0.561. Như vậy model yếu nhất với xe ở xa, chỉ còn
chấm đèn, gần đường chân trời. Ngay cả xe lớn gần camera cũng chỉ bắt được khoảng một nửa.

Trong `compare_round0.jpg`, các box vàng (bỏ sót) tập trung ở: cụm xe nhỏ ở đoạn đường xa phía trên
(frame_0150, frame_0350) và xe gần camera bị chói đèn pha (xe góc dưới trái frame_0250 và frame_0050).
Các box đỏ (báo nhầm) chủ yếu là box chồng lên cụm hai xe sát nhau ở làn xa (frame_0150 bên phải,
frame_0250 giữa ảnh).

Trường hợp cần rà lại nhãn tham chiếu: ở frame_0350, bộ tham chiếu có một box nhỏ ở góc trên bên trái,
ngang vùng biển quảng cáo phát sáng phía trên mặt đường, không giống xe. Nếu đó là nhãn sai, model bị
tính một FN oan. Tôi ghi nhận ca này thay vì sửa `data/test/labels/`.

## 3. Chiến lược chọn mẫu

`score = W_U·U + W_A·A + W_D·D` với trọng số 0.5 / 0.3 / 0.2:

- **U (bất định):** với mỗi box model dự đoán có conf c, `u = 1 − |2c − 1|`, bằng 1 khi c = 0.5 (model
  phân vân nhất). U là trung bình 5 giá trị u lớn nhất của frame.
- **A (mơ hồ):** số box có 0.15 ≤ c < 0.50, chia cho số lớn nhất trong pool. Frame nhiều box "nửa tin nửa
  ngờ" được ưu tiên.
- **D (đa dạng thời gian):** khoảng cách tới frame đã gán gần nhất, chặn ở 10 s. Vòng 1 chưa có frame nào
  đã gán nên D = 1 cho mọi frame.
- **`MIN_GAP_S = 2 s`:** khi chọn tham lam theo score, bỏ frame cách frame đã chọn dưới 2 s, vì camera
  đứng yên nên hai frame quá gần gần như trùng nhau; gán cả hai tốn công mà model học thêm rất ít.

Theo `reports/SELECTION.md`: frame_0182 (hạng 1, A = 1.0) phải thêm 14 box; frame_0369 (hạng 2, 43 box ứng
viên nhưng pre-label chỉ 14) phải thêm 26 box; frame_0331 (hạng 5, A = 1.0) vừa xóa 6 box giả vừa thêm 21.
Như vậy điểm cao đúng là chỉ vào ảnh nhiều lỗi pre-label. Frame_0372 (hạng 6) bị loại vì chỉ cách
frame_0369 1.2 s, đúng với luật khoảng cách. Tuy vậy `MIN_GAP_S` vẫn để lọt cặp frame_0326/0331 và
frame_0182/0187 (đều cách 2.0 s), làm tăng chi phí gán mà thêm ít thông tin.

**Điểm bất định không chứng minh ảnh đó sẽ cải thiện model.** U chỉ đo trên box model đã dự đoán; xe bị bỏ
sót hoàn toàn không sinh box nên vô hình với U và A, trong khi bỏ sót là lỗi lớn nhất (phải thêm 175 box).
Thực tế sau khi train trên lô "khó" này, AP50 trên test lại giảm (mục 4).

## 4. Các vòng học chủ động (active learning)

Bảng từ `rounds_table.md` (20 ảnh test, 403 box, IoU 0.5, P/R/F1 tại conf 0.25):

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 327 | 0.537 | -0.235 | 1.000 | 0.151 | 0.263 | 0.000 | 0.162 | 0.317 |

**Mức sửa nhãn vòng 1** (`outputs/round1_diff.md`): model đề xuất 169 box trên 12 ảnh; tôi giữ 148
(accepted), chỉnh 4 (edited), xóa 17 box giả (deleted), thêm 175 box bị bỏ sót (added); tổng 327 box.
Accept rate 88% nghe cao, nhưng chỉ 152/327 box cuối (46%) đến từ pre-label; hơn một nửa phải thêm tay.
Con số này khớp với recall 0.489 của cold start trên test: lỗi chủ yếu của nhãn AI là bỏ sót, không phải
báo nhầm. Ảnh sửa nhiều nhất: frame_0369 (thêm 26), frame_0331 (xóa 6, thêm 21), frame_0326 (thêm 19).

**Model đã train đúng lô đã sửa:** `metrics_round1.json` ghi `n_train_images = 12`, `n_train_boxes = 327`,
đúng bằng số box cuối trong `round1_diff.md`; 50 epoch, imgsz 960, Tesla T4.

**Kết quả trên cùng 20 ảnh test:** AP50 giảm 0.235 (0.771 → 0.537), vượt xa ngưỡng nhiễu khoảng 0.01, nên đây
là thay đổi thật. Tại conf 0.25: TP 197 → 61, FP 16 → 0, FN 206 → 342. Model sau fine-tune **không còn báo
nhầm nhưng bỏ sót nặng hơn** ở mọi nhóm kích thước: xe nhỏ 0.182 → 0.000, trung bình 0.547 → 0.162, lớn
0.561 → 0.317. AP50 (tính trên mọi ngưỡng conf) vẫn còn 0.537 trong khi recall tại 0.25 chỉ 0.151, cho thấy
nhiều dự đoán đúng vẫn còn nhưng có độ tin cậy dưới 0.25. Model trở nên quá thận trọng.

**Ca thay đổi trong `compare_round1.jpg`:**

- *Tốt hơn:* frame_0350, xe gần camera ở góc dưới bên trái bị cold start bỏ sót (vàng) nhưng model vòng 1
  bắt được (xanh); box đỏ lớn báo nhầm ở bên trái của cold start cũng biến mất.
- *Xấu hơn:* frame_0150, xe gần camera ở góc dưới bên trái (xe lớn, rõ) được cold start bắt (xanh) nhưng
  vòng 1 bỏ sót (vàng); TP của ảnh này giảm từ 10 xuống 3.

Lý do có thể kiểm: model gốc học 3 lớp COCO (car/bus/truck); fine-tune về 1 lớp `car` làm đầu phân loại
học lại từ đầu, chỉ với 12 ảnh cùng một góc camera. Độ tin cậy dự đoán bị kéo thấp, phần lớn rơi dưới
ngưỡng 0.25.

**Phân biệt ba nguồn bằng chứng:**

- *Quan sát độc lập* (`BLIND_SCAN.md`, khóa trước khi xem pre-label): frame_0099 tôi đếm được 24 xe, dự đoán
  AI dễ bỏ sót cụm 3 xe đèn đỏ ở xa gần chân trời (làn phải) và gộp hai xe con sát nhau ở làn thứ 2 từ trái.
- *Lỗi pre-label đã sửa* (`round1_diff.md`, `REVIEW_LOG.csv`): frame_0099 pre-label chỉ có 13 box; sau khi
  sửa còn 25 box (giữ 12, chỉnh 1, thêm 12), gần với 24 xe tôi đếm lúc quét độc lập. Trong `REVIEW_LOG.csv`:
  AI bỏ sót cả xe con lớn, rõ nhất ảnh ở giữa đáy frame_0099 (added), không chỉ xe xa; cụm xe nhỏ gần chân
  trời mà BLIND_SCAN dự đoán đúng là bị bỏ sót (added); ở frame_0331, một box rộng 116 px gộp hai xe đi cạnh
  nhau bị xóa để vẽ hai box riêng (deleted). Lưu ý: 6 box "deleted" ở frame_0331 không phải đều là vật giả;
  một số là box bị vẽ lại quá khác (IoU < 0.5) nên được đếm là xóa + thêm, ví dụ xe nhòe ở mép dưới phải
  (65×88 px → 183×106 px).
- *Kết quả model sau train* (`compare_round1.jpg`, `metrics_round1.json`): như phân tích ở trên.

**Ca khó theo guideline:** xe ở rất xa gần chân trời, chỉ còn chấm đèn, box cao khoảng 15–18 px (frame_0099,
cy ≈ 284–295 px). Guideline cho phép gán hoặc không với box dưới 16 px vì chúng bị bỏ qua khi chấm. Tôi chọn
**gán** khi còn phân biệt được cặp đèn của từng xe, vẽ box theo phần thân đoán được quanh cụm đèn chứ không
chỉ khoanh chấm đèn, và giữ cách này cho cả 12 ảnh. Hệ quả: lô train có nhiều box nhỏ (27 box/ảnh so với
khoảng 20 box/ảnh trong tham chiếu), một khác biệt kiểu gán có thể góp phần vào việc model vòng 1 khớp kém
với bộ tham chiếu.

## 5. Kết luận và giới hạn

**So với cold start:** vòng 1 kém hơn rõ trên cùng test (AP50 0.771 → 0.537, recall 0.489 → 0.151), dù
precision tăng lên 1.0. Nhãn đã sửa có chất lượng tốt hơn nhãn AI (thêm 175 xe bị bỏ sót, xóa 17 box giả),
nhưng nhãn tốt hơn chưa chuyển thành model tốt hơn: vấn đề nằm ở cách fine-tune với quá ít ảnh, không phải
ở việc chọn ảnh.

**Quyết định: dừng gán thêm nhãn ở vòng 2 cho tới khi kiểm tra xong quy trình train.** Gán thêm 12 ảnh
(khoảng 175 box phải thêm tay mỗi lô) vào một quy trình đang làm model tệ đi là lãng phí công. Trước khi
train thêm, tôi sẽ kiểm tra:

1. Đánh giá model vòng 1 ở ngưỡng conf thấp hơn (0.05–0.1) hoặc xem đường PR, để xác nhận model vẫn tìm
   được xe nhưng tự tin thấp.
2. Cấu hình fine-tune: đóng băng backbone, giảm learning rate, hoặc giữ đầu COCO rồi gộp car/bus/truck
   thay vì học lại đầu 1 lớp từ 12 ảnh.
3. Tính nhất quán nhãn: tôi đã thêm nhiều xe nhỏ ở xa; nếu các box dưới 16 px không nhất quán giữa các ảnh,
   model học phải nhiễu.

**Hai ca còn yếu cho vòng sau:**

1. *Xe nhỏ ở xa gần chân trời:* recall xe nhỏ còn 0.000 ở vòng 1 (0.182 ở cold start). Chi phí rà cao vì
   mỗi ảnh đông có hàng chục xe nhỏ (frame_0369 phải thêm 26 box), và một phần nằm dưới 16 px nên không được
   tính khi chấm.
2. *Xe lớn gần camera bị chói đèn pha:* recall xe lớn giảm 0.561 → 0.317 (ví dụ frame_0150 góc dưới trái).
   Ca này rẻ để gán (ít xe, box rõ) nhưng các frame gần nhau gần như trùng lặp, nên cần tăng `MIN_GAP_S`
   lên khoảng 4–5 s để lô không chứa cùng một đợt xe.

**Giới hạn ảnh hưởng tới kết luận:**

- Tập test chỉ 20 ảnh từ 4 đoạn thời gian; chênh lệch nhỏ dưới 0.01 AP50 không đủ để kết luận. Mức giảm
  0.235 vượt xa ngưỡng này, nhưng vẫn chỉ đại diện cho 4 đoạn video.
- 14 box tham chiếu dưới 16 px bị bỏ qua, nên số đo không phản ánh xe ở rất xa, đúng nhóm model yếu nhất.
- Nhãn tham chiếu do một model khác tạo, chưa có người rà (ví dụ box nghi là biển quảng cáo ở frame_0350).
  Model được fine-tune theo nhãn người sửa (27 box/ảnh, so với 20 box/ảnh trong tham chiếu) có thể bị phạt
  khi hai kiểu gán khác nhau.
- Chỉ có một vòng, một lô 12 ảnh và không có đối chứng chọn ngẫu nhiên, nên chưa so được uncertainty
  sampling với chọn ngẫu nhiên.

**Tự QC:** `check_submission.py` kiểm hình thức; hash nhãn test khớp bản phát hành; `BLIND_SCAN.md` khóa
trước khi mở pre-label và chưa đổi sau khi khóa (`blind_lock.json`); số box train trong `metrics_round1.json`
(327) khớp `round1_diff.md`.
