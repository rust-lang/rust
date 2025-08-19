//@only-target: linux android illumos
// test_race, test_blocking_read and test_blocking_write depend on a deterministic schedule.
//@compile-flags: -Zmiri-deterministic-concurrency

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

use std::thread;

fn main() {
    test_read_write();
    test_race();

    #[cfg(not(target_os = "illumos"))]
    test_syscall();

    test_blocking_read();
    test_blocking_write();
    test_two_threads_blocked_on_eventfd();
}

fn read_bytes<const N: usize>(fd: i32, buf: &mut [u8; N]) -> i32 {
    let res: i32 = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), N).try_into().unwrap() };
    return res;
}

fn write_bytes<const N: usize>(fd: i32, data: [u8; N]) -> i32 {
    let res: i32 =
        unsafe { libc::write(fd, data.as_ptr() as *const libc::c_void, N).try_into().unwrap() };
    return res;
}

fn test_read_write() {
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };
    let sized_8_data: [u8; 8] = 1_u64.to_ne_bytes();
    // Write 1 to the counter.
    let res = write_bytes(fd, sized_8_data);
    assert_eq!(res, 8);

    // Read 1 from the counter.
    let mut buf: [u8; 8] = [0; 8];
    let res = read_bytes(fd, &mut buf);
    // Read returns number of bytes has been read, which is always 8.
    assert_eq!(res, 8);
    // Check the value of counter read.
    let counter = u64::from_ne_bytes(buf);
    assert_eq!(counter, 1);

    // After read, the counter is currently 0, read counter 0 should fail with return
    // value -1.
    let mut buf: [u8; 8] = [0; 8];
    let res = read_bytes(fd, &mut buf);
    let e = std::io::Error::last_os_error();
    assert_eq!(e.raw_os_error(), Some(libc::EAGAIN));
    assert_eq!(res, -1);

    // Write with supplied buffer bigger than 8 bytes should be allowed.
    let sized_9_data: [u8; 9];
    if cfg!(target_endian = "big") {
        // Adjust the data based on the endianness of host system.
        sized_9_data = [0, 0, 0, 0, 0, 0, 0, 1, 0];
    } else {
        sized_9_data = [1, 0, 0, 0, 0, 0, 0, 0, 0];
    }
    let res = write_bytes(fd, sized_9_data);
    assert_eq!(res, 8);

    // Read with supplied buffer smaller than 8 bytes should fail with return
    // value -1.
    let mut buf: [u8; 7] = [1; 7];
    let res = read_bytes(fd, &mut buf);
    let e = std::io::Error::last_os_error();
    assert_eq!(e.raw_os_error(), Some(libc::EINVAL));
    assert_eq!(res, -1);

    // Write with supplied buffer smaller than 8 bytes should fail with return
    // value -1.
    let size_7_data: [u8; 7] = [1; 7];
    let res = write_bytes(fd, size_7_data);
    let e = std::io::Error::last_os_error();
    assert_eq!(e.raw_os_error(), Some(libc::EINVAL));
    assert_eq!(res, -1);

    // Read with supplied buffer bigger than 8 bytes should be allowed.
    let mut buf: [u8; 9] = [1; 9];
    let res = read_bytes(fd, &mut buf);
    assert_eq!(res, 8);

    // Write u64::MAX should fail.
    let u64_max_bytes: [u8; 8] = [255; 8];
    let res = write_bytes(fd, u64_max_bytes);
    let e = std::io::Error::last_os_error();
    assert_eq!(e.raw_os_error(), Some(libc::EINVAL));
    assert_eq!(res, -1);
}

fn test_race() {
    static mut VAL: u8 = 0;
    let flags = libc::EFD_NONBLOCK | libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };
    let thread1 = thread::spawn(move || {
        let mut buf: [u8; 8] = [0; 8];
        let res = read_bytes(fd, &mut buf);
        // read returns number of bytes has been read, which is always 8.
        assert_eq!(res, 8);
        let counter = u64::from_ne_bytes(buf);
        assert_eq!(counter, 1);
        unsafe { assert_eq!(VAL, 1) };
    });
    unsafe { VAL = 1 };
    let data: [u8; 8] = 1_u64.to_ne_bytes();
    let res = write_bytes(fd, data);
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
    let flags = libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };
    let thread1 = thread::spawn(move || {
        let mut buf: [u8; 8] = [0; 8];
        // This will block.
        let res = read_bytes(fd, &mut buf);
        // read returns number of bytes has been read, which is always 8.
        assert_eq!(res, 8);
        let counter = u64::from_ne_bytes(buf);
        assert_eq!(counter, 1);
    });
    let sized_8_data: [u8; 8] = 1_u64.to_ne_bytes();
    // Pass control to thread1 so it can block on eventfd `read`.
    thread::yield_now();
    // Write 1 to the counter to unblock thread1.
    let res = write_bytes(fd, sized_8_data);
    assert_eq!(res, 8);
    thread1.join().unwrap();
}

/// This test will block on eventfd `write` then get unblocked by `read`.
fn test_blocking_write() {
    // eventfd write will block when EFD_NONBLOCK flag is clear
    // and the addition caused counter to exceed u64::MAX - 1.
    let flags = libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };
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
    let res = read_bytes(fd, &mut buf);
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
    let flags = libc::EFD_CLOEXEC;
    let fd = unsafe { libc::eventfd(0, flags) };
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
        let res = read_bytes(fd, &mut buf);
        // read returns number of bytes has been read, which is always 8.
        assert_eq!(res, 8);
        let counter = u64::from_ne_bytes(buf);
        assert_eq!(counter, (u64::MAX - 1));
    });

    thread1.join().unwrap();
    thread2.join().unwrap();
    thread3.join().unwrap();
}
