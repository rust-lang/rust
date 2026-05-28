use crate::os::wasi::io::RawFd;

#[test]
fn test_raw_fd_layout() {
    // `OwnedFd` and `BorrowedFd` use pattern types with ranges that depend on
    // the bit width of `RawFd`. If this ever changes, those values will need
    // to be updated.
    assert_eq!(size_of::<RawFd>(), 4);
}
