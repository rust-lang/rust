//@ignore-target: windows # No libc socketpair on Windows
//~^ERROR: deadlocked
//~^^ERROR: deadlocked
// test_race depends on a deterministic schedule.
//@compile-flags: -Zmiri-deterministic-concurrency
//@error-in-other-file: deadlock
//@require-annotations-for-level: error

use std::thread;

#[path = "../../utils/libc.rs"]
mod libc_utils;

// Test the behaviour of a thread being blocked on write, get unblocked, then blocked again.

// The expected execution is
// 1. Thread 1 blocks.
// 2. Thread 2 blocks.
// 3. Thread 3 unblocks both thread 1 and thread 2.
// 4. Thread 1 writes.
// 5. Thread 2's `write` can never complete -> deadlocked.
fn main() {
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);
    let arr1: [u8; 212992] = [1; 212992];
    // Exhaust the space in the buffer so the subsequent write will block.
    let res =
        unsafe { libc_utils::write_all(fds[0], arr1.as_ptr() as *const libc::c_void, 212992) };
    assert_eq!(res, 212992);
    let thread1 = thread::spawn(move || {
        let data = "a".as_bytes();
        // The write below will be blocked because the buffer is already full.
        let res = unsafe { libc::write(fds[0], data.as_ptr() as *const libc::c_void, data.len()) };
        assert_eq!(res, data.len().cast_signed());
    });
    let thread2 = thread::spawn(move || {
        let data = "a".as_bytes();
        // The write below will be blocked because the buffer is already full.
        let res = unsafe { libc::write(fds[0], data.as_ptr() as *const libc::c_void, data.len()) };
        //~^ERROR: deadlock
        assert_eq!(res, data.len().cast_signed());
    });
    let thread3 = thread::spawn(move || {
        // Unblock thread1 by freeing up some space.
        let mut buf: [u8; 1] = [0; 1];
        let res = unsafe { libc::read(fds[1], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) };
        assert_eq!(res, buf.len().cast_signed());
        assert_eq!(buf, [1]);
    });
    thread1.join().unwrap();
    thread2.join().unwrap();
    thread3.join().unwrap();
}
