//@only-target: linux android illumos
//~^ERROR: deadlocked
//~^^ERROR: deadlocked
//@compile-flags: -Zmiri-deterministic-concurrency
//@error-in-other-file: deadlock

use std::thread;

// Test the behaviour of a thread being blocked on an eventfd read, get unblocked, and then
// get blocked again.

// The expected execution is
// 1. Thread 1 blocks.
// 2. Thread 2 blocks.
// 3. Thread 3 unblocks both thread 1 and thread 2.
// 4. Thread 1 reads.
// 5. Thread 2's `read` can never complete -> deadlocked.

fn main() {
    // eventfd write will block when EFD_NONBLOCK flag is clear
    // and the addition caused counter to exceed u64::MAX - 1.
    let flags = libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };

    let thread1 = thread::spawn(move || {
        let mut buf: [u8; 8] = [0; 8];
        // This read will block initially.
        let res: i64 = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), 8).try_into().unwrap() };
        assert_eq!(res, 8);
        let counter = u64::from_ne_bytes(buf);
        assert_eq!(counter, 1_u64);
    });

    let thread2 = thread::spawn(move || {
        let mut buf: [u8; 8] = [0; 8];
        // This read will block initially, then get unblocked by thread3, then get blocked again
        // because the `read` in thread1 executes first and set the counter to 0 again.
        let res: i64 = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), 8).try_into().unwrap() };
        //~^ERROR: deadlocked
        assert_eq!(res, 8);
        let counter = u64::from_ne_bytes(buf);
        assert_eq!(counter, 1_u64);
    });

    let thread3 = thread::spawn(move || {
        let sized_8_data = 1_u64.to_ne_bytes();
        // Write 1 to the counter, so both thread1 and thread2 will unblock.
        let res: i64 = unsafe {
            libc::write(fd, sized_8_data.as_ptr() as *const libc::c_void, 8).try_into().unwrap()
        };
        // Make sure that write is successful.
        assert_eq!(res, 8);
    });

    thread1.join().unwrap();
    thread2.join().unwrap();
    thread3.join().unwrap();
}
