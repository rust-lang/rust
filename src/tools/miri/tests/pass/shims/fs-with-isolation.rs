//@ignore-target: windows # File handling is not implemented yet
//@compile-flags: -Zmiri-isolation-error=warn-nobacktrace
//@normalize-stderr-test: "(stat(x)?)" -> "$$STAT"

use std::fs::{self, File};
use std::io::ErrorKind;
use std::os::unix;

fn main() {
    // test `open`
    assert_eq!(File::create("foo.txt").unwrap_err().kind(), ErrorKind::PermissionDenied);

    // test `unlink`
    assert_eq!(fs::remove_file("foo.txt").unwrap_err().kind(), ErrorKind::PermissionDenied);

    // test `symlink`
    assert_eq!(
        unix::fs::symlink("foo.txt", "foo_link.txt").unwrap_err().kind(),
        ErrorKind::PermissionDenied
    );

    // test `stat`
    assert_eq!(fs::metadata("foo.txt").unwrap_err().kind(), ErrorKind::PermissionDenied);

    // test `rename`
    assert_eq!(fs::rename("a.txt", "b.txt").unwrap_err().kind(), ErrorKind::PermissionDenied);

    // test `mkdir`
    assert_eq!(fs::create_dir("foo/bar").unwrap_err().kind(), ErrorKind::PermissionDenied);

    // test `rmdir`
    assert_eq!(fs::remove_dir("foo/bar").unwrap_err().kind(), ErrorKind::PermissionDenied);

    // test `opendir`
    assert_eq!(fs::read_dir("foo/bar").unwrap_err().kind(), ErrorKind::PermissionDenied);
}
