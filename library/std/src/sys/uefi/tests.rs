use super::alloc::*;

#[test]
fn align() {
    // UEFI ABI specifies that allocation alignment minimum is always 8. So this can be
    // statically verified.
    assert_eq!(POOL_ALIGNMENT, 8);

    // Loop over allocation-request sizes from 0-256 and alignments from 1-128, and verify
    // that in case of overalignment there is at least space for one additional pointer to
    // store in the allocation.
    for i in 0..256 {
        for j in &[1, 2, 4, 8, 16, 32, 64, 128] {
            if *j <= 8 {
                assert_eq!(align_size(i, *j), i);
            } else {
                assert!(align_size(i, *j) > i + std::mem::size_of::<*mut ()>());
            }
        }
    }
}
