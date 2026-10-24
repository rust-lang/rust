//@compile-flags: -Zmiri-isolation-error=warn-nobacktrace
//@normalize-stderr-test: "(getcwd|GetCurrentDirectoryW)" -> "$$GETCWD"
//@normalize-stderr-test: "(chdir|SetCurrentDirectoryW)" -> "$$SETCWD"

use std::env;
use std::io::ErrorKind;

fn main() {
    // Test that current dir operations return a proper error instead
    // of stopping the machine in isolation mode
    assert_eq!(env::current_dir().unwrap_err().kind(), ErrorKind::PermissionDenied);
    for _i in 0..3 {
        // Ensure we get no repeated warnings when doing this multiple times.
        assert_eq!(env::current_dir().unwrap_err().kind(), ErrorKind::PermissionDenied);
    }

    assert_eq!(env::set_current_dir("..").unwrap_err().kind(), ErrorKind::PermissionDenied);
    for _i in 0..3 {
        assert_eq!(env::set_current_dir("..").unwrap_err().kind(), ErrorKind::PermissionDenied);
    }
}
