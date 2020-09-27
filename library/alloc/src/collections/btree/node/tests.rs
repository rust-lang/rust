use super::*;

#[test]
fn test_splitpoint() {
    for idx in 0..=CAPACITY {
        let (middle_kv_idx, insertion) = splitpoint(idx);

        // Simulate performing the split:
        let mut left_len = middle_kv_idx;
        let mut right_len = CAPACITY - middle_kv_idx - 1;
        match insertion {
            InsertionPlace::Left(edge_idx) => {
                assert!(edge_idx <= left_len);
                left_len += 1;
            }
            InsertionPlace::Right(edge_idx) => {
                assert!(edge_idx <= right_len);
                right_len += 1;
            }
        }
        assert!(left_len >= MIN_LEN);
        assert!(right_len >= MIN_LEN);
        assert!(left_len + right_len == CAPACITY);
    }
}
