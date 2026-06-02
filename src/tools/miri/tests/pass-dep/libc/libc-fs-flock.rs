//@ignore-target: windows # no libc
//@ignore-target: solaris # Does not have flock
//@compile-flags: -Zmiri-disable-isolation

//@revisions: windows_host unix_host
//@[unix_host] ignore-host: windows
//@[windows_host] only-host: windows

use std::fs::File;
use std::os::fd::AsRawFd;

#[path = "../../utils/libc.rs"]
mod libc_utils;
#[path = "../../utils/mod.rs"]
mod utils;
use libc_utils::*;

fn main() {
    let bytes = b"Hello, World!\n";
    let path = utils::prepare_with_content("miri_test_fs_shared_lock.txt", bytes);

    let files: Vec<File> = (0..3).map(|_| File::open(&path).unwrap()).collect();

    // Test that we can apply many shared locks.
    for file in files.iter() {
        errno_check(unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_SH) });
    }

    // Test that shared lock prevents exclusive lock.
    {
        let fd = files[0].as_raw_fd();
        let err =
            errno_result(unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) }).unwrap_err();
        assert_eq!(err.raw_os_error().unwrap(), libc::EWOULDBLOCK);
    }

    // Unlock shared lock.
    for file in files.iter() {
        errno_check(unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) });
    }

    // Take exclusive lock.
    {
        let fd = files[0].as_raw_fd();
        errno_check(unsafe { libc::flock(fd, libc::LOCK_EX) });
    }

    // Test that exclusive lock prevents exclusive and shared locks.
    {
        let fd = files[1].as_raw_fd();
        let err =
            errno_result(unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) }).unwrap_err();
        assert_eq!(err.raw_os_error().unwrap(), libc::EWOULDBLOCK);

        let fd = files[2].as_raw_fd();
        let err =
            errno_result(unsafe { libc::flock(fd, libc::LOCK_SH | libc::LOCK_NB) }).unwrap_err();
        assert_eq!(err.raw_os_error().unwrap(), libc::EWOULDBLOCK);
    }

    // Unlock exclusive lock.
    {
        let fd = files[0].as_raw_fd();
        errno_check(unsafe { libc::flock(fd, libc::LOCK_UN) });
        // Redundant unlock also works.
        // FIXME(#miri/5074): except on Windows hosts...
        if !cfg!(windows_host) {
            errno_check(unsafe { libc::flock(fd, libc::LOCK_UN) });
        }
    }

    // Test behavior when we acquire multiple locks on the same FD.
    // FIXME(#miri/5074): this does not behave correctly on Windows hosts.
    if !cfg!(windows_host) {
        let fd1 = files[1].as_raw_fd();
        let fd2 = files[2].as_raw_fd();

        errno_check(unsafe { libc::flock(fd1, libc::LOCK_EX | libc::LOCK_NB) });
        // This converts the exclusive lock to a shared lock.
        errno_check(unsafe { libc::flock(fd1, libc::LOCK_SH | libc::LOCK_NB) });
        // Now the other fd can have the shared lock as well.
        errno_check(unsafe { libc::flock(fd2, libc::LOCK_SH | libc::LOCK_NB) });

        // Reset.
        errno_check(unsafe { libc::flock(fd1, libc::LOCK_UN) });
        errno_check(unsafe { libc::flock(fd2, libc::LOCK_UN) });

        // Getting first a shared lock and then upgrading to exclusive should also work.
        errno_check(unsafe { libc::flock(fd1, libc::LOCK_SH | libc::LOCK_NB) });
        errno_check(unsafe { libc::flock(fd1, libc::LOCK_EX | libc::LOCK_NB) });
        // This is truly exclusive: fd2 is locked out.
        let err =
            errno_result(unsafe { libc::flock(fd2, libc::LOCK_SH | libc::LOCK_NB) }).unwrap_err();
        assert_eq!(err.raw_os_error().unwrap(), libc::EWOULDBLOCK);
    }
}
