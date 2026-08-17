//@only-target: linux android illumos
//@compile-flags: -Zmiri-deterministic-concurrency
//@error-in-other-file: deadlock

use std::thread;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::*;

// Test the behaviour of a thread being blocked on an eventfd `write`, get unblocked, and then
// get blocked again.

// The expected execution is
// 1. Thread 1 blocks.
// 2. Thread 2 blocks.
// 3. Thread 3 unblocks both thread 1 and thread 2.
// 4. Thread 1 writes u64::MAX.
// 5. Thread 2's `write` can never complete -> deadlocked.
fn main() {
    // eventfd write will block when EFD_NONBLOCK flag is clear
    // and the addition caused counter to exceed u64::MAX - 1.
    let flags = libc::EFD_CLOEXEC;
    let fd = errno_result(unsafe { libc::eventfd(0, flags) }).unwrap();
    // Write u64::MAX - 1, so the all subsequent write will block.
    eventfd::write_val(fd, u64::MAX - 1).unwrap();

    let thread1 = thread::spawn(move || {
        eventfd::write_val(fd, u64::MAX - 1).unwrap();
    });

    let thread2 = thread::spawn(move || {
        // The thread will be briefly unblocked internally, but immediately block again since
        // it still cannot write.
        eventfd::write_val(fd, u64::MAX - 1).unwrap();
        //~^ERROR: deadlocked
    });

    let thread3 = thread::spawn(move || {
        // This will unblock both `write` in thread1 and thread2.
        let val = eventfd::read_val(fd).unwrap();
        assert_eq!(val, u64::MAX - 1);
    });

    thread1.join().unwrap();
    thread2.join().unwrap();
    thread3.join().unwrap();
}
