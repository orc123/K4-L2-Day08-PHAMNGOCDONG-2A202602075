# Vì sao chọn lô này?

Nguồn: `outputs/selection_round1.csv` (268 frame pool, xếp theo `score = 0.5·U + 0.3·A + 0.2·D`),
`outputs/selection_round1.jpg`, và `outputs/round1_diff.json` (kết quả sau khi sửa, dùng để đối chiếu).
Vòng 1 chưa có frame nào đã gán nên D = 1.0 cho mọi frame; thứ hạng do U và A quyết định.

## Top 5 nếu chỉ đủ công rà năm ảnh

Trong 50 dòng đầu, không frame nào có `empty = True` (model luôn dự đoán ít nhất một box), nên
thưởng cho frame trống không tác động. Quyết định chính là tránh ảnh gần trùng: camera đứng yên,
xe ở lại trong khung hình vài giây, nên hai frame cách nhau 2–5 giây thường chứa cùng dòng xe.
Tôi ưu tiên trải đều trên trục thời gian.

| Thứ tự | Frame | Hạng CSV | t (s) | score | U | A | n_boxes | Lý do |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | frame_0182.jpg | 1 | 72.8 | 0.9591 | 0.918 | 1.000 | 28 | Điểm cao nhất; số box mơ hồ đạt max pool (18). |
| 2 | frame_0369.jpg | 2 | 147.6 | 0.9324 | 0.932 | 0.889 | 43 | Đoạn cuối video, đường đông; 43 box ứng viên (conf ≥ 0.05). |
| 3 | frame_0331.jpg | 5 | 132.4 | 0.9154 | 0.831 | 1.000 | 47 | A = 1.0, nhiều box ứng viên; chọn thay cho frame_0326. |
| 4 | frame_0099.jpg | 8 | 39.6 | 0.9063 | 0.946 | 0.778 | 29 | U cao nhất trong top 10, phủ đoạn đầu video (trước 40 s). |
| 5 | frame_0227.jpg | 11 | 90.8 | 0.8915 | 0.916 | 0.778 | 37 | Lấp khoảng 73–132 s; contact sheet có xe tải/xe lớn phát sáng. |

Hai frame bỏ qua dù xếp hạng cao hơn:

- **frame_0380** (hạng 3, score 0.917, t = 152.0 s): chỉ cách frame_0369 4.4 s, cùng một đợt xe đông ở
  cuối video. Với ngân sách năm ảnh, gán cả hai tốn công mà thêm ít thông tin mới.
- **frame_0326** (hạng 4, score 0.9155, t = 130.4 s): cách frame_0331 đúng 2.0 s, vừa bằng `MIN_GAP_S`,
  nên lô 12 ảnh vẫn nhận cả hai. Với năm ảnh, tôi chỉ giữ frame_0331 vì A = 1.0 và nhiều box hơn (47).

## Ba frame thuộc lô 12 ảnh và bằng chứng

- **frame_0182** (hạng 1): U = 0.918, A = 1.0 (18 box có conf 0.15–0.50, nhiều nhất pool). Sau khi sửa:
  pre-label 13 box → cuối 26 box (thêm 14, xóa 1). Model phân vân đúng ở ảnh mà nó bỏ sót gần một nửa số xe.
- **frame_0369** (hạng 2): U = 0.932, 43 box ứng viên ở conf ≥ 0.05 nhưng pre-label chỉ có 14 box. Sau khi
  sửa còn 39 box (thêm 26, nhiều nhất lô). Nhiều ứng viên conf thấp thực ra là xe thật; contact sheet cho
  thấy dòng xe dày ở các làn xa.
- **frame_0331** (hạng 5): A = 1.0, 47 box ứng viên. Đây là ảnh sửa nhiều nhất theo cả hai hướng: xóa 6 box
  giả, thêm 21 box (pre-label 20 → cuối 35). Điểm mơ hồ cao ứng với cả bỏ sót lẫn báo nhầm.

## Một frame điểm cao nhưng không chọn, và một frame điểm thấp vẫn nên xem

- **frame_0372** (hạng 6, score 0.9101, cao hơn 7 frame được chọn): bị loại vì cách frame_0369 chỉ 1.2 s
  (< `MIN_GAP_S` = 2 s). Tương tự frame_0368 (hạng 9, cách 0.4 s) và frame_0330 (hạng 12, cách frame_0331
  0.4 s). Tôi đồng ý loại: hai frame cách nhau dưới 1.2 s gần như là cùng một ảnh.
- **frame_0195** (hạng 268, thấp nhất, score 0.5721, U = 0.644, chỉ 3 box mơ hồ): điểm thấp vì model tự tin
  với các box nó thấy. Nhưng lỗi chính của model là **bỏ sót** (175/327 box cuối phải thêm tay), và xe bị bỏ
  sót hoàn toàn không sinh box nên không góp vào U hay A. Một ảnh "tự tin" vẫn có thể thiếu nhiều xe;
  nên rà ngẫu nhiên vài frame điểm thấp làm đối chứng.

## Điều phép chọn này chưa chứng minh về chất lượng mô hình

- Điểm bất định chỉ đo trên box model **đã dự đoán**; nó không thấy xe bị bỏ sót hoàn toàn.
- Điểm cao cho thấy model phân vân, không chứng minh ảnh đó sẽ làm model tốt lên. Sau khi fine-tune trên lô
  này, AP50 trên test giảm từ 0.771 xuống 0.537 (`reports/rounds_table.md`), nên việc chọn đúng ảnh khó chưa
  đủ để cải thiện model.
- `MIN_GAP_S = 2 s` vẫn để lọt cặp gần trùng (frame_0326/0331 và frame_0182/0187, đều cách 2.0 s).
- Chưa có đối chứng chọn ngẫu nhiên (`STRATEGY = "random"`), nên chưa biết uncertainty sampling có tốt hơn
  chọn ngẫu nhiên cùng ngân sách hay không.
