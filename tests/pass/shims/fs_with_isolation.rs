//@ignore-target-windows: File handling is not implemented yet
//@compile-flags: -Zmiri-isolation-error=warn-nobacktrace
//@normalize-stderr-test: "(stat(x)?)" -> "$$STAT"

use std::ffi::CString;
use std::fs::{self, File};
use std::io::{Error, ErrorKind};
use std::os::unix;

fn main() {
    // test `open`
    assert_eq!(File::create("foo.txt").unwrap_err().kind(), ErrorKind::PermissionDenied);

    // test `fcntl`
    unsafe {
        assert_eq!(libc::fcntl(1, libc::F_DUPFD, 0), -1);
        assert_eq!(Error::last_os_error().raw_os_error(), Some(libc::EPERM));
    }

    // test `unlink`
    assert_eq!(fs::remove_file("foo.txt").unwrap_err().kind(), ErrorKind::PermissionDenied);

    // test `symlink`
    assert_eq!(
        unix::fs::symlink("foo.txt", "foo_link.txt").unwrap_err().kind(),
        ErrorKind::PermissionDenied
    );

    // test `readlink`
    let symlink_c_str = CString::new("foo.txt").unwrap();
    let mut buf = vec![0; "foo_link.txt".len() + 1];
    unsafe {
        assert_eq!(libc::readlink(symlink_c_str.as_ptr(), buf.as_mut_ptr(), buf.len()), -1);
        assert_eq!(Error::last_os_error().raw_os_error(), Some(libc::EACCES));
    }

    // test `stat`
    assert_eq!(fs::metadata("foo.txt").unwrap_err().kind(), ErrorKind::PermissionDenied);
    assert_eq!(Error::last_os_error().raw_os_error(), Some(libc::EACCES));

    // test `rename`
    assert_eq!(fs::rename("a.txt", "b.txt").unwrap_err().kind(), ErrorKind::PermissionDenied);

    // test `mkdir`
    assert_eq!(fs::create_dir("foo/bar").unwrap_err().kind(), ErrorKind::PermissionDenied);

    // test `rmdir`
    assert_eq!(fs::remove_dir("foo/bar").unwrap_err().kind(), ErrorKind::PermissionDenied);

    // test `opendir`
    assert_eq!(fs::read_dir("foo/bar").unwrap_err().kind(), ErrorKind::PermissionDenied);
    assert_eq!(Error::last_os_error().raw_os_error(), Some(libc::EACCES));
}
