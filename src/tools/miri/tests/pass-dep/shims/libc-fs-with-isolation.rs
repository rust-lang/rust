//@ignore-target-windows: no libc on Windows
//@compile-flags: -Zmiri-isolation-error=warn-nobacktrace
//@normalize-stderr-test: "(stat(x)?)" -> "$$STAT"

use std::ffi::CString;
use std::fs;
use std::io::{Error, ErrorKind};

fn main() {
    // test `fcntl(F_DUPFD): should work even with isolation.`
    unsafe {
        assert!(libc::fcntl(1, libc::F_DUPFD, 0) >= 0);
    }

    // test `readlink`
    let symlink_c_str = CString::new("foo.txt").unwrap();
    let mut buf = vec![0; "foo_link.txt".len() + 1];
    unsafe {
        assert_eq!(libc::readlink(symlink_c_str.as_ptr(), buf.as_mut_ptr(), buf.len()), -1);
        assert_eq!(Error::last_os_error().raw_os_error(), Some(libc::EACCES));
    }

    // test `stat`
    let err = fs::metadata("foo.txt").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::PermissionDenied);
    // check that it is the right kind of `PermissionDenied`
    assert_eq!(err.raw_os_error(), Some(libc::EACCES));
}
