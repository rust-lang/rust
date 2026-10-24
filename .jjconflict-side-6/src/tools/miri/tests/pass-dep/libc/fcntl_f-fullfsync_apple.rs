//@only-target: apple # F_FULLFSYNC only on apple systems
//@compile-flags: -Zmiri-isolation-error=warn-nobacktrace

use std::io::Error;

fn main() {
    // test `fcntl(F_FULLFSYNC)`
    unsafe {
        assert_eq!(libc::fcntl(1, libc::F_FULLFSYNC, 0), -1);
        assert_eq!(Error::last_os_error().raw_os_error(), Some(libc::EPERM));
    }
}
