//@ignore-target: windows # No libc socketpair on Windows
//~^ERROR: deadlocked
//~^^ERROR: deadlocked
// test_race depends on a deterministic schedule.
//@compile-flags: -Zmiri-deterministic-concurrency
//@error-in-other-file: deadlock

use std::thread;

// Test the behaviour of a thread being blocked on read, get unblocked, then blocked again.

// The expected execution is
// 1. Thread 1 blocks.
// 2. Thread 2 blocks.
// 3. Thread 3 unblocks both thread 1 and thread 2.
// 4. Thread 1 reads.
// 5. Thread 2's `read` can never complete -> deadlocked.

fn main() {
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);
    let thread1 = thread::spawn(move || {
        // Let this thread block on read.
        let mut buf: [u8; 3] = [0; 3];
        let res = unsafe { libc::read(fds[1], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) };
        assert_eq!(res, 3);
        assert_eq!(&buf, "abc".as_bytes());
    });
    let thread2 = thread::spawn(move || {
        // Let this thread block on read.
        let mut buf: [u8; 3] = [0; 3];
        let res = unsafe { libc::read(fds[1], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) };
        //~^ERROR: deadlocked
        assert_eq!(res, 3);
        assert_eq!(&buf, "abc".as_bytes());
    });
    let thread3 = thread::spawn(move || {
        // Unblock thread1 by writing something.
        let data = "abc".as_bytes().as_ptr();
        let res = unsafe { libc::write(fds[0], data as *const libc::c_void, 3) };
        assert_eq!(res, 3);
    });
    thread1.join().unwrap();
    thread2.join().unwrap();
    thread3.join().unwrap();
}
