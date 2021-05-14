// compile-flags: -Zmiri-isolation-error=warn-nobacktrace
// normalize-stderr-test "(getcwd|GetCurrentDirectoryW)" -> "$$GET"
// normalize-stderr-test "(chdir|SetCurrentDirectoryW)" -> "$$SET"

use std::env;
use std::io::ErrorKind;

fn main() {
    // Test that current dir operations return a proper error instead
    // of stopping the machine in isolation mode
    assert_eq!(env::current_dir().unwrap_err().kind(), ErrorKind::NotFound);
    for _i in 0..3 {
        assert_eq!(env::current_dir().unwrap_err().kind(), ErrorKind::NotFound);
    }

    assert_eq!(env::set_current_dir("..").unwrap_err().kind(), ErrorKind::NotFound);
    for _i in 0..3 {
        assert_eq!(env::set_current_dir("..").unwrap_err().kind(), ErrorKind::NotFound);
    }
}
