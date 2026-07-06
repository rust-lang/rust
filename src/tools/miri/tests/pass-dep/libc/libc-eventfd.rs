//@only-target: linux android illumos
// test_race, test_blocking_read and test_blocking_write depend on a deterministic schedule.
//@compile-flags: -Zmiri-deterministic-concurrency
//@run-native

// FIXME(static_mut_refs): use raw pointers instead of references
#![allow(static_mut_refs)]

use std::{io, thread};

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
}

// We want to do individual read/write calls here so we avoid read_exact/write_all.

fn read_bytes<const N: usize>(fd: i32, buf: &mut [u8; N]) -> io::Result<usize> {
    libc_utils::errno_result(unsafe { libc::read(fd, buf.as_mut_ptr().cast(), N) })
        .map(|len| len.try_into().unwrap())
}

fn write_bytes<const N: usize>(fd: i32, data: [u8; N]) -> io::Result<usize> {
    libc_utils::errno_result(unsafe { libc::write(fd, data.as_ptr() as *const libc::c_void, N) })
        .map(|len| len.try_into().unwrap())
}

fn test_read_write() {
    let fd =
        errno_result(unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) }).unwrap();
    let sized_8_data: [u8; 8] = 1_u64.to_ne_bytes();
    // Write 1 to the counter.
    let res = write_bytes(fd, sized_8_data).unwrap();
    assert_eq!(res, 8);
    // Write 0 to the counter,
    let res = write_bytes(fd, [0u8; 8]).unwrap();
    assert_eq!(res, 8);

    // Read 1 from the counter.
    let mut buf: [u8; 8] = [0; 8];
    let res = read_bytes(fd, &mut buf).unwrap();
    // Read returns number of bytes has been read, which is always 8.
    assert_eq!(res, 8);
    // Check the value of counter read.
    assert_eq!(u64::from_ne_bytes(buf), 1);

    // After read, the counter is currently 0, read counter 0 should fail with EAGAIN.
    let mut buf: [u8; 8] = [0; 8];
    let err = read_bytes(fd, &mut buf).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EAGAIN));

    // Write with supplied buffer bigger than 8 bytes should be allowed according to the docs,
    // but tests on real systems indicate that it fails.
    let sized_9_data = [0u8; 9];
    let err = write_bytes(fd, sized_9_data).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));

    // Read with supplied buffer smaller than 8 bytes should fail with return
    // value -1.
    let mut buf: [u8; 7] = [1; 7];
    let err = read_bytes(fd, &mut buf).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));

    // Write with supplied buffer smaller than 8 bytes should fail with return
    // value -1.
    let size_7_data: [u8; 7] = [1; 7];
    let err = write_bytes(fd, size_7_data).unwrap_err();
    assert_eq!(err.raw_os_error(), Some(libc::EINVAL));

    // Do two valid writes so we have something to read for the next test.
    let res = write_bytes(fd, sized_8_data).unwrap();
    assert_eq!(res, 8);
    let res = write_bytes(fd, sized_8_data).unwrap();
    assert_eq!(res, 8);

    // Read with supplied buffer bigger than 8 bytes should be allowed.
    let mut buf: [u8; 9] = [1; 9];
    let res = read_bytes(fd, &mut buf).unwrap();
    assert_eq!(res, 8);
    let buf: &[u8; 8] = (&buf[..8]).try_into().unwrap();
    assert_eq!(u64::from_ne_bytes(*buf), 2);

    // Write u64::MAX should fail.
    let u64_max_bytes: [u8; 8] = [255; 8];
    let err = write_bytes(fd, u64_max_bytes).unwrap_err();
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

        let mut buf: [u8; 8] = [0; 8];
        let res = read_bytes(fd, &mut buf).unwrap();
        // read returns number of bytes has been read, which is always 8.
        assert_eq!(res, 8);
        let counter = u64::from_ne_bytes(buf);
        assert_eq!(counter, 1);
        unsafe { assert_eq!(VAL, 1) };
    });
    unsafe { VAL = 1 };
    let data: [u8; 8] = 1_u64.to_ne_bytes();
    let res = write_bytes(fd, data).unwrap();
    // write returns number of bytes written, which is always 8.
    assert_eq!(res, 8);
    thread::yield_now();
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
        let mut buf: [u8; 8] = [0; 8];
        // This will block.
        let res = read_bytes(fd, &mut buf).unwrap();
        // read returns number of bytes has been read, which is always 8.
        assert_eq!(res, 8);
        let counter = u64::from_ne_bytes(buf);
        assert_eq!(counter, 1);
    });
    let sized_8_data: [u8; 8] = 1_u64.to_ne_bytes();
    // Pass control to thread1 so it can block on eventfd `read`.
    thread::yield_now();
    // Write 1 to the counter to unblock thread1.
    let res = write_bytes(fd, sized_8_data).unwrap();
    assert_eq!(res, 8);
    thread1.join().unwrap();
}

/// This test will block on eventfd `write` then get unblocked by `read`.
fn test_blocking_write() {
    // eventfd write will block when EFD_NONBLOCK flag is clear
    // and the addition caused counter to exceed u64::MAX - 1.
    let fd = errno_result(unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) }).unwrap();
    // Write u64 - 1, so the all subsequent write will block.
    let sized_8_data: [u8; 8] = (u64::MAX - 1).to_ne_bytes();
    let res: i64 = unsafe {
        libc::write(fd, sized_8_data.as_ptr() as *const libc::c_void, 8).try_into().unwrap()
    };
    assert_eq!(res, 8);

    let thread1 = thread::spawn(move || {
        let sized_8_data = 1_u64.to_ne_bytes();
        // Write 1 to the counter, this will block.
        let res: i64 = unsafe {
            libc::write(fd, sized_8_data.as_ptr() as *const libc::c_void, 8).try_into().unwrap()
        };
        // Make sure that write is successful.
        assert_eq!(res, 8);
    });
    let mut buf: [u8; 8] = [0; 8];
    // Pass control to thread1 so it can block on eventfd `write`.
    thread::yield_now();
    // This will unblock previously blocked eventfd read.
    let res = read_bytes(fd, &mut buf).unwrap();
    // read returns number of bytes has been read, which is always 8.
    assert_eq!(res, 8);
    let counter = u64::from_ne_bytes(buf);
    assert_eq!(counter, (u64::MAX - 1));
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
    // Write u64 - 1, so the all subsequent write will block.
    let sized_8_data: [u8; 8] = (u64::MAX - 1).to_ne_bytes();
    let res: i64 = unsafe {
        libc::write(fd, sized_8_data.as_ptr() as *const libc::c_void, 8).try_into().unwrap()
    };
    assert_eq!(res, 8);

    let thread1 = thread::spawn(move || {
        let sized_8_data = 1_u64.to_ne_bytes();
        let res: i64 = unsafe {
            libc::write(fd, sized_8_data.as_ptr() as *const libc::c_void, 8).try_into().unwrap()
        };
        // Make sure that write is successful.
        assert_eq!(res, 8);
    });

    let thread2 = thread::spawn(move || {
        let sized_8_data = 1_u64.to_ne_bytes();
        let res: i64 = unsafe {
            libc::write(fd, sized_8_data.as_ptr() as *const libc::c_void, 8).try_into().unwrap()
        };
        // Make sure that write is successful.
        assert_eq!(res, 8);
    });

    let thread3 = thread::spawn(move || {
        let mut buf: [u8; 8] = [0; 8];
        // This will unblock previously blocked eventfd read.
        let res = read_bytes(fd, &mut buf).unwrap();
        // read returns number of bytes has been read, which is always 8.
        assert_eq!(res, 8);
        let counter = u64::from_ne_bytes(buf);
        assert_eq!(counter, (u64::MAX - 1));
    });

    thread1.join().unwrap();
    thread2.join().unwrap();
    thread3.join().unwrap();
}
