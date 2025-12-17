//@ignore-target: windows # No libc socketpair on Windows
// test_race depends on a deterministic schedule.
//@compile-flags: -Zmiri-deterministic-concurrency

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

use std::thread;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::*;

fn main() {
    test_socketpair();
    test_socketpair_threaded();
    test_race();
    test_blocking_read();
    test_blocking_write();
    test_socketpair_setfl_getfl();
}

fn test_socketpair() {
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Read size == data available in buffer.
    let data = b"abcde";
    write_all_from_slice(fds[0], data).unwrap();
    let buf = read_all_into_array::<5>(fds[1]).unwrap();
    assert_eq!(&buf, data);

    // Read size > data available in buffer.
    let data = b"abc";
    write_all_from_slice(fds[0], data).unwrap();
    let mut buf2: [u8; 5] = [0; 5];
    let (read, rest) = read_into_slice(fds[1], &mut buf2).unwrap();
    assert_eq!(read[..], data[..read.len()]);
    // Write 2 more bytes so we can exactly fill the `rest`.
    write_all_from_slice(fds[0], b"12").unwrap();
    read_all_into_slice(fds[1], rest).unwrap();

    // Test read and write from another direction.
    // Read size == data available in buffer.
    let data = b"12345";
    write_all_from_slice(fds[1], data).unwrap();
    let buf3 = read_all_into_array::<5>(fds[0]).unwrap();
    assert_eq!(&buf3, data);

    // Read size > data available in buffer.
    let data = b"123";
    write_all_from_slice(fds[1], data).unwrap();
    let mut buf4: [u8; 5] = [0; 5];
    let (read, rest) = read_into_slice(fds[0], &mut buf4).unwrap();
    assert_eq!(read[..], data[..read.len()]);
    // Write 2 more bytes so we can exactly fill the `rest`.
    write_all_from_slice(fds[1], b"12").unwrap();
    read_all_into_slice(fds[0], rest).unwrap();

    // Test when happens when we close one end, with some data in the buffer.
    write_all_from_slice(fds[0], data).unwrap();
    errno_check(unsafe { libc::close(fds[0]) });
    // Reading the other end should return that data, then EOF.
    let mut buf: [u8; 5] = [0; 5];
    read_all_into_slice(fds[1], &mut buf[0..3]).unwrap();
    assert_eq!(&buf[0..3], data);
    let res = read_into_slice(fds[1], &mut buf[3..5]).unwrap().0.len();
    assert_eq!(res, 0); // 0-sized read: EOF.
    // Writing the other end should emit EPIPE.
    let res = write_all_from_slice(fds[1], &mut buf);
    assert_eq!(res, Err(-1));
    assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(libc::EPIPE));
}

fn test_socketpair_threaded() {
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    let thread1 = thread::spawn(move || {
        let buf = read_all_into_array::<5>(fds[1]).unwrap();
        assert_eq!(&buf, b"abcde");
    });
    thread::yield_now();
    write_all_from_slice(fds[0], b"abcde").unwrap();
    thread1.join().unwrap();

    // Read and write from different direction
    let thread2 = thread::spawn(move || {
        thread::yield_now();
        write_all_from_slice(fds[1], b"12345").unwrap();
    });
    let buf = read_all_into_array::<5>(fds[0]).unwrap();
    assert_eq!(&buf, b"12345");
    thread2.join().unwrap();
}

fn test_race() {
    static mut VAL: u8 = 0;
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });
    let thread1 = thread::spawn(move || {
        // write() from the main thread will occur before the read() here
        // because preemption is disabled and the main thread yields after write().
        let buf = read_all_into_array::<1>(fds[1]).unwrap();
        assert_eq!(&buf, b"a");
        // The read above establishes a happens-before so it is now safe to access this global variable.
        unsafe { assert_eq!(VAL, 1) };
    });
    unsafe { VAL = 1 };
    write_all_from_slice(fds[0], b"a").unwrap();
    thread::yield_now();
    thread1.join().unwrap();
}

// Test the behaviour of a socketpair getting blocked on read and subsequently unblocked.
fn test_blocking_read() {
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });
    let thread1 = thread::spawn(move || {
        // Let this thread block on read.
        let buf = read_all_into_array::<3>(fds[1]).unwrap();
        assert_eq!(&buf, b"abc");
    });
    let thread2 = thread::spawn(move || {
        // Unblock thread1 by doing writing something.
        write_all_from_slice(fds[0], b"abc").unwrap();
    });
    thread1.join().unwrap();
    thread2.join().unwrap();
}

// Test the behaviour of a socketpair getting blocked on write and subsequently unblocked.
fn test_blocking_write() {
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });
    let arr1: [u8; 0x34000] = [1; 0x34000];
    // Exhaust the space in the buffer so the subsequent write will block.
    write_all_from_slice(fds[0], &arr1).unwrap();
    let thread1 = thread::spawn(move || {
        // The write below will be blocked because the buffer is already full.
        write_all_from_slice(fds[0], b"abc").unwrap();
    });
    let thread2 = thread::spawn(move || {
        // Unblock thread1 by freeing up some space.
        let buf = read_all_into_array::<3>(fds[1]).unwrap();
        assert_eq!(buf, [1, 1, 1]);
    });
    thread1.join().unwrap();
    thread2.join().unwrap();
}

/// Basic test for socketpair fcntl's F_SETFL and F_GETFL flag.
fn test_socketpair_setfl_getfl() {
    // Initialise socketpair fds.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) });

    // Test if both sides have O_RDWR.
    assert_eq!(errno_result(unsafe { libc::fcntl(fds[0], libc::F_GETFL) }).unwrap(), libc::O_RDWR);
    assert_eq!(errno_result(unsafe { libc::fcntl(fds[1], libc::F_GETFL) }).unwrap(), libc::O_RDWR);

    // Add the O_NONBLOCK flag with F_SETFL.
    errno_check(unsafe { libc::fcntl(fds[0], libc::F_SETFL, libc::O_NONBLOCK) });

    // Test if the O_NONBLOCK flag is successfully added.
    assert_eq!(
        errno_result(unsafe { libc::fcntl(fds[0], libc::F_GETFL) }).unwrap(),
        libc::O_RDWR | libc::O_NONBLOCK
    );

    // The other side remains unchanged.
    assert_eq!(errno_result(unsafe { libc::fcntl(fds[1], libc::F_GETFL) }).unwrap(), libc::O_RDWR);

    // Test if O_NONBLOCK flag can be unset.
    errno_check(unsafe { libc::fcntl(fds[0], libc::F_SETFL, 0) });
    assert_eq!(errno_result(unsafe { libc::fcntl(fds[0], libc::F_GETFL) }).unwrap(), libc::O_RDWR);
}
