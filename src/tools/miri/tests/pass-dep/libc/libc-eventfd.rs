//@only-target: linux android illumos
// test_race, test_blocking_read and test_blocking_write depend on a deterministic schedule.
//@compile-flags: -Zmiri-deterministic-concurrency
//@run-native

// FIXME(static_mut_refs): use raw pointers instead of references
#![allow(static_mut_refs)]

use std::thread;
use std::time::Duration;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::*;

fn main() {
    test_read_write();
    test_race();

    #[cfg(not(target_os = "illumos"))]
    test_syscall();

    test_blocking_read();
    test_blocking_write();
    test_two_threads_blocked_on_eventfd();
    test_close_while_blocked();
}

// We want to do individual read/write calls here so we avoid read_exact/write_all.

fn test_read_write() {
    let fd =
        errno_result(unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) }).unwrap();

    // Write 1 to the counter.
    eventfd::write_val(fd, 1).unwrap();
    // Write 0 to the counter.
    eventfd::write_val(fd, 0).unwrap();

    // Read 1 from the counter.
    let val = eventfd::read_val(fd).unwrap();
    assert_eq!(val, 1);

    // After read, the counter is currently 0, read counter 0 should fail with EAGAIN.
    let err = eventfd::read_val(fd).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EAGAIN));

    // Write with supplied buffer bigger than 8 bytes should be allowed according to the docs,
    // but tests on real systems indicate that it fails.
    let err = write_all(fd, &[0u8; 9]).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));

    // Read with supplied buffer smaller than 8 bytes should fail.
    let err = read_exact_array::<7>(fd).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));

    // Write with supplied buffer smaller than 8 bytes should fail.
    let err = write_all(fd, &[0u8; 9]).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));

    // Do valid write so we have something to read for the next test.
    eventfd::write_val(fd, 2).unwrap();

    // Read with supplied buffer bigger than 8 bytes should be allowed.
    let mut buf: [u8; 9] = [1; 9];
    let len = errno_result(unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) }).unwrap();
    assert_eq!(len, 8);
    assert_eq!(u64::from_ne_bytes(buf[..8].try_into().unwrap()), 2);

    // Write u64::MAX should fail.
    let err = eventfd::write_val(fd, u64::MAX).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));
}

fn test_race() {
    static mut VAL: u8 = 0;

    let fd =
        errno_result(unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) }).unwrap();
    let thread1 = thread::spawn(move || {
        if !cfg!(miri) {
            // Make sure the write goes first.
            thread::sleep(std::time::Duration::from_millis(10));
        }

        let val = eventfd::read_val(fd).unwrap();
        assert_eq!(val, 1);
        unsafe { assert_eq!(VAL, 1) };
    });
    unsafe { VAL = 1 };
    eventfd::write_val(fd, 1).unwrap();
    thread1.join().unwrap();
}

// This is a test for calling eventfd2 through a syscall.
// Illumos supports eventfd, but it has no entry to call it through syscall.
#[cfg(not(target_os = "illumos"))]
fn test_syscall() {
    let initval = 0 as libc::c_uint;
    let flags = (libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) as libc::c_int;
    let fd = unsafe { libc::syscall(libc::SYS_eventfd2, initval, flags) };
    assert_ne!(fd, -1);
}

// This test will block on eventfd read then get unblocked by `write`.
fn test_blocking_read() {
    // eventfd read will block when EFD_NONBLOCK flag is clear and counter = 0.
    let fd = errno_result(unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) }).unwrap();
    let thread1 = thread::spawn(move || {
        let val = eventfd::read_val(fd).unwrap();
        assert_eq!(val, 1);
    });
    // Pass control to thread1 so it can block on eventfd `read`.
    thread::yield_now();
    // Write 1 to the counter to unblock thread1.
    eventfd::write_val(fd, 1).unwrap();
    thread1.join().unwrap();
}

/// This test will block on eventfd `write` then get unblocked by `read`.
fn test_blocking_write() {
    // eventfd write will block when EFD_NONBLOCK flag is clear
    // and the addition caused counter to exceed u64::MAX - 1.
    let fd = errno_result(unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) }).unwrap();
    // Write u64::MAX - 1, so the all subsequent write will block.
    eventfd::write_val(fd, u64::MAX - 1).unwrap();

    let thread1 = thread::spawn(move || {
        // Write 1 to the counter, this will block.
        eventfd::write_val(fd, 1).unwrap();
    });
    // Pass control to thread1 so it can block on eventfd `write`.
    thread::yield_now();
    // This will unblock previously blocked eventfd read.
    let val = eventfd::read_val(fd).unwrap();
    assert_eq!(val, u64::MAX - 1);
    thread1.join().unwrap();
}

// Test two threads blocked on eventfd.
// Expected behaviour:
// 1. thread1 and thread2 both blocked on `write`.
// 2. thread3 unblocks both thread1 and thread2
// 3. The write in thread1 and thread2 return successfully.
fn test_two_threads_blocked_on_eventfd() {
    // eventfd write will block when EFD_NONBLOCK flag is clear
    // and the addition caused counter to exceed u64::MAX - 1.
    let fd = errno_result(unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) }).unwrap();
    // Write u64::MAX - 1, so the all subsequent write will block.
    eventfd::write_val(fd, u64::MAX - 1).unwrap();

    let thread1 = thread::spawn(move || {
        eventfd::write_val(fd, 1).unwrap();
    });

    let thread2 = thread::spawn(move || {
        eventfd::write_val(fd, 1).unwrap();
    });

    let thread3 = thread::spawn(move || {
        let val = eventfd::read_val(fd).unwrap();
        assert_eq!(val, u64::MAX - 1);
    });

    thread1.join().unwrap();
    thread2.join().unwrap();
    thread3.join().unwrap();
}

/// Test what happens when we close the eventfd file *descriptor* while a thread
/// is blocked on it. We have to keep opemn the file *description* since otherwise
/// we can never unblock the thread again.
fn test_close_while_blocked() {
    let fd = errno_result(unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) }).unwrap();
    let fd2 = errno_result(unsafe { libc::dup(fd) }).unwrap();

    // Spawn server thread.
    let server_thread = thread::spawn(move || {
        if !cfg!(miri) {
            // Ensure main thread is blocked on reading from the client socket.
            thread::sleep(Duration::from_millis(10));
        }

        unsafe { errno_check(libc::close(fd)) };

        eventfd::write_val(fd2, 1).unwrap();
    });

    let val = eventfd::read_val(fd).unwrap();
    assert_eq!(val, 1);

    server_thread.join().unwrap();
}
