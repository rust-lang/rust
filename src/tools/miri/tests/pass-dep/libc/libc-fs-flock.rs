//@ignore-target: windows # File handling is not implemented yet
//@ignore-target: solaris # Does not have flock
//@compile-flags: -Zmiri-disable-isolation

use std::fs::File;
use std::io::Error;
use std::os::fd::AsRawFd;

#[path = "../../utils/mod.rs"]
mod utils;

fn main() {
    let bytes = b"Hello, World!\n";
    let path = utils::prepare_with_content("miri_test_fs_shared_lock.txt", bytes);

    let files: Vec<File> = (0..3).map(|_| File::open(&path).unwrap()).collect();

    // Test that we can apply many shared locks
    for file in files.iter() {
        let fd = file.as_raw_fd();
        let ret = unsafe { libc::flock(fd, libc::LOCK_SH) };
        if ret != 0 {
            panic!("flock error: {}", Error::last_os_error());
        }
    }

    // Test that shared lock prevents exclusive lock
    {
        let fd = files[0].as_raw_fd();
        let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(ret, -1);
        let err = Error::last_os_error().raw_os_error().unwrap();
        assert_eq!(err, libc::EWOULDBLOCK);
    }

    // Unlock shared lock
    for file in files.iter() {
        let fd = file.as_raw_fd();
        let ret = unsafe { libc::flock(fd, libc::LOCK_UN) };
        if ret != 0 {
            panic!("flock error: {}", Error::last_os_error());
        }
    }

    // Take exclusive lock
    {
        let fd = files[0].as_raw_fd();
        let ret = unsafe { libc::flock(fd, libc::LOCK_EX) };
        assert_eq!(ret, 0);
    }

    // Test that shared lock prevents exclusive and shared locks
    {
        let fd = files[1].as_raw_fd();
        let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        assert_eq!(ret, -1);
        let err = Error::last_os_error().raw_os_error().unwrap();
        assert_eq!(err, libc::EWOULDBLOCK);

        let fd = files[2].as_raw_fd();
        let ret = unsafe { libc::flock(fd, libc::LOCK_SH | libc::LOCK_NB) };
        assert_eq!(ret, -1);
        let err = Error::last_os_error().raw_os_error().unwrap();
        assert_eq!(err, libc::EWOULDBLOCK);
    }

    // Unlock exclusive lock
    {
        let fd = files[0].as_raw_fd();
        let ret = unsafe { libc::flock(fd, libc::LOCK_UN) };
        assert_eq!(ret, 0);
    }
}
