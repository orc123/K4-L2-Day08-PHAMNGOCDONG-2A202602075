# Quét độc lập trước khi xem pre-label

Frame: frame_0099.jpg

Số xe nhìn thấy bằng mắt: 24

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:

1. Phần trên ảnh, cuối đường gần chân trời, làn phải, chiều đi: 3 xe ở rất xa, chỉ còn thấy các chấm đèn đỏ, thân xe gần như không phân biệt được → box rất nhỏ nên AI dễ bỏ sót.
2. Giữa ảnh, làn thứ 2 từ trái, đoạn xa trên làn đông xe: hai xe con đi sát nhau, đèn/thân xe chồng lên nhau → AI dễ gộp thành 1 box hoặc bỏ sót xe phía sau.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
