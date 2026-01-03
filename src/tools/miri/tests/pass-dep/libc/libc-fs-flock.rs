//@ignore-target: windows # File handling is not implemented yet
//@ignore-target: solaris # Does not have flock
//@compile-flags: -Zmiri-disable-isolation

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

    // Test that we can apply many shared locks
    for file in files.iter() {
        errno_check(unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_SH) });
    }

    // Test that shared lock prevents exclusive lock
    {
        let fd = files[0].as_raw_fd();
        let err =
            errno_result(unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) }).unwrap_err();
        assert_eq!(err.raw_os_error().unwrap(), libc::EWOULDBLOCK);
    }

    // Unlock shared lock
    for file in files.iter() {
        errno_check(unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) });
    }

    // Take exclusive lock
    {
        let fd = files[0].as_raw_fd();
        errno_check(unsafe { libc::flock(fd, libc::LOCK_EX) });
    }

    // Test that shared lock prevents exclusive and shared locks
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

    // Unlock exclusive lock
    {
        let fd = files[0].as_raw_fd();
        errno_check(unsafe { libc::flock(fd, libc::LOCK_UN) });
    }
}
