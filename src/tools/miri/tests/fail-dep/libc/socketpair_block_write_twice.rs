//@ignore-target: windows # No libc socketpair on Windows
//@compile-flags: -Zmiri-deterministic-concurrency
//@error-in-other-file: deadlock
//@require-annotations-for-level: error

use std::thread;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::{errno_check, errno_result, read_exact_array, write_all};

// Test the behaviour of a thread being blocked on write, get unblocked, then blocked again.

// The expected execution is
// 1. Thread 1 blocks.
// 2. Thread 2 blocks.
// 3. Thread 3 unblocks both thread 1 and thread 2.
// 4. Thread 1 writes.
// 5. Thread 2's `write` can never complete -> deadlocked.
fn main() {
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });
    let arr1: [u8; 212992] = [1; 212992];
    // Exhaust the space in the buffer so the subsequent write will block.
    write_all(fds[0], &arr1).unwrap();
    let thread1 = thread::spawn(move || {
        // The write below will be blocked because the buffer is already full.
        write_all(fds[0], b"a").unwrap();
    });
    let thread2 = thread::spawn(move || {
        let data = "a".as_bytes();
        // The write below will be blocked because the buffer is already full.
        let res = errno_result(unsafe {
            libc::write(fds[0], data.as_ptr() as *const libc::c_void, data.len())
            //~^ERROR: deadlock
        })
        .unwrap();
        assert_eq!(res, data.len().cast_signed());
    });
    let thread3 = thread::spawn(move || {
        // Unblock thread1 by freeing up some space.
        let buf = read_exact_array::<1>(fds[1]).unwrap();
        assert_eq!(buf, [1]);
    });
    thread1.join().unwrap();
    thread2.join().unwrap();
    thread3.join().unwrap();
}
