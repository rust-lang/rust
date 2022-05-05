#[cfg(any(unix, target_os = "wasi"))]
#[test]
fn test_raw_fd() {
    #[cfg(unix)]
    use crate::os::unix::io::{AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, RawFd};
    #[cfg(target_os = "wasi")]
    use crate::os::wasi::io::{AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, RawFd};

    let raw_fd: RawFd = crate::io::stdin().as_raw_fd();

    let stdin_as_file = unsafe { crate::fs::File::from_raw_fd(raw_fd) };
    assert_eq!(stdin_as_file.as_raw_fd(), raw_fd);
    assert_eq!(unsafe { BorrowedFd::borrow_raw(raw_fd).as_raw_fd() }, raw_fd);
    assert_eq!(stdin_as_file.into_raw_fd(), 0);
}

#[cfg(any(unix, target_os = "wasi"))]
#[test]
fn test_fd() {
    #[cfg(unix)]
    use crate::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
    #[cfg(target_os = "wasi")]
    use crate::os::wasi::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};

    let stdin = crate::io::stdin();
    let fd: BorrowedFd<'_> = stdin.as_fd();
    let raw_fd: RawFd = fd.as_raw_fd();
    let owned_fd: OwnedFd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

    let stdin_as_file = crate::fs::File::from(owned_fd);

    assert_eq!(stdin_as_file.as_fd().as_raw_fd(), raw_fd);
    assert_eq!(Into::<OwnedFd>::into(stdin_as_file).into_raw_fd(), raw_fd);
}
