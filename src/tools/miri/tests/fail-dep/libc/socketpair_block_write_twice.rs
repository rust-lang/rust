//@ignore-target: windows # No libc socketpair on Windows
//~^ERROR: deadlocked
//~^^ERROR: deadlocked
// test_race depends on a deterministic schedule.
//@compile-flags: -Zmiri-deterministic-concurrency
//@error-in-other-file: deadlock

use std::thread;

// Test the behaviour of a thread being blocked on write, get unblocked, then blocked again.

// The expected execution is
// 1. Thread 1 blocks.
// 2. Thread 2 blocks.
// 3. Thread 3 unblocks both thread 1 and thread 2.
// 4. Thread 1 reads.
// 5. Thread 2's `write` can never complete -> deadlocked.
fn main() {
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);
    let arr1: [u8; 212992] = [1; 212992];
    // Exhaust the space in the buffer so the subsequent write will block.
    let res = unsafe { libc::write(fds[0], arr1.as_ptr() as *const libc::c_void, 212992) };
    assert_eq!(res, 212992);
    let thread1 = thread::spawn(move || {
        let data = "abc".as_bytes().as_ptr();
        // The write below will be blocked because the buffer is already full.
        let res = unsafe { libc::write(fds[0], data as *const libc::c_void, 3) };
        assert_eq!(res, 3);
    });
    let thread2 = thread::spawn(move || {
        let data = "abc".as_bytes().as_ptr();
        // The write below will be blocked because the buffer is already full.
        let res = unsafe { libc::write(fds[0], data as *const libc::c_void, 3) };
        //~^ERROR: deadlocked
        assert_eq!(res, 3);
    });
    let thread3 = thread::spawn(move || {
        // Unblock thread1 by freeing up some space.
        let mut buf: [u8; 3] = [0; 3];
        let res = unsafe { libc::read(fds[1], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) };
        assert_eq!(res, 3);
        assert_eq!(buf, [1, 1, 1]);
    });
    thread1.join().unwrap();
    thread2.join().unwrap();
    thread3.join().unwrap();
}
