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

#[test]
#[cfg(target_arch = "x86_64")]
fn test_sizes() {
    assert_eq!(core::mem::size_of::<LeafNode<(), ()>>(), 16);
    assert_eq!(core::mem::size_of::<LeafNode<i64, i64>>(), 16 + CAPACITY * 8 * 2);
    assert_eq!(core::mem::size_of::<InternalNode<(), ()>>(), 112);
    assert_eq!(core::mem::size_of::<InternalNode<i64, i64>>(), 112 + CAPACITY * 8 * 2);
}
