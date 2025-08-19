//@ignore-target: windows # No libc socketpair on Windows
// test_race depends on a deterministic schedule.
//@compile-flags: -Zmiri-deterministic-concurrency

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

use std::thread;

#[path = "../../utils/libc.rs"]
mod libc_utils;

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
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Read size == data available in buffer.
    let data = "abcde".as_bytes().as_ptr();
    let res = unsafe { libc_utils::write_all(fds[0], data as *const libc::c_void, 5) };
    assert_eq!(res, 5);
    let mut buf: [u8; 5] = [0; 5];
    let res =
        unsafe { libc_utils::read_all(fds[1], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) };
    assert_eq!(res, 5);
    assert_eq!(buf, "abcde".as_bytes());

    // Read size > data available in buffer.
    let data = "abc".as_bytes();
    let res = unsafe { libc_utils::write_all(fds[0], data.as_ptr() as *const libc::c_void, 3) };
    assert_eq!(res, 3);
    let mut buf2: [u8; 5] = [0; 5];
    let res = unsafe { libc::read(fds[1], buf2.as_mut_ptr().cast(), buf2.len() as libc::size_t) };
    assert!(res > 0 && res <= 3);
    let res = res as usize;
    assert_eq!(buf2[..res], data[..res]);
    if res < 3 {
        // Drain the rest from the read end.
        let res = unsafe { libc_utils::read_all(fds[1], buf2[res..].as_mut_ptr().cast(), 3 - res) };
        assert!(res > 0);
    }

    // Test read and write from another direction.
    // Read size == data available in buffer.
    let data = "12345".as_bytes().as_ptr();
    let res = unsafe { libc_utils::write_all(fds[1], data as *const libc::c_void, 5) };
    assert_eq!(res, 5);
    let mut buf3: [u8; 5] = [0; 5];
    let res = unsafe {
        libc_utils::read_all(fds[0], buf3.as_mut_ptr().cast(), buf3.len() as libc::size_t)
    };
    assert_eq!(res, 5);
    assert_eq!(buf3, "12345".as_bytes());

    // Read size > data available in buffer.
    let data = "123".as_bytes();
    let res = unsafe { libc_utils::write_all(fds[1], data.as_ptr() as *const libc::c_void, 3) };
    assert_eq!(res, 3);
    let mut buf4: [u8; 5] = [0; 5];
    let res = unsafe { libc::read(fds[0], buf4.as_mut_ptr().cast(), buf4.len() as libc::size_t) };
    assert!(res > 0 && res <= 3);
    let res = res as usize;
    assert_eq!(buf4[..res], data[..res]);
    if res < 3 {
        // Drain the rest from the read end.
        let res = unsafe { libc_utils::read_all(fds[0], buf4[res..].as_mut_ptr().cast(), 3 - res) };
        assert!(res > 0);
    }

    // Test when happens when we close one end, with some data in the buffer.
    let res = unsafe { libc_utils::write_all(fds[0], data.as_ptr() as *const libc::c_void, 3) };
    assert_eq!(res, 3);
    unsafe { libc::close(fds[0]) };
    // Reading the other end should return that data, then EOF.
    let mut buf: [u8; 5] = [0; 5];
    let res =
        unsafe { libc_utils::read_all(fds[1], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) };
    assert_eq!(res, 3);
    assert_eq!(&buf[0..3], "123".as_bytes());
    let res =
        unsafe { libc_utils::read_all(fds[1], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) };
    assert_eq!(res, 0); // 0-sized read: EOF.
    // Writing the other end should emit EPIPE.
    let res = unsafe { libc_utils::write_all(fds[1], data.as_ptr() as *const libc::c_void, 1) };
    assert_eq!(res, -1);
    assert_eq!(std::io::Error::last_os_error().raw_os_error(), Some(libc::EPIPE));
}

fn test_socketpair_threaded() {
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    let thread1 = thread::spawn(move || {
        let mut buf: [u8; 5] = [0; 5];
        let res: i64 = unsafe {
            libc_utils::read_all(fds[1], buf.as_mut_ptr().cast(), buf.len() as libc::size_t)
                .try_into()
                .unwrap()
        };
        assert_eq!(res, 5);
        assert_eq!(buf, "abcde".as_bytes());
    });
    thread::yield_now();
    let data = "abcde".as_bytes().as_ptr();
    let res = unsafe { libc_utils::write_all(fds[0], data as *const libc::c_void, 5) };
    assert_eq!(res, 5);
    thread1.join().unwrap();

    // Read and write from different direction
    let thread2 = thread::spawn(move || {
        thread::yield_now();
        let data = "12345".as_bytes().as_ptr();
        let res = unsafe { libc_utils::write_all(fds[1], data as *const libc::c_void, 5) };
        assert_eq!(res, 5);
    });
    let mut buf: [u8; 5] = [0; 5];
    let res =
        unsafe { libc_utils::read_all(fds[0], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) };
    assert_eq!(res, 5);
    assert_eq!(buf, "12345".as_bytes());
    thread2.join().unwrap();
}

fn test_race() {
    static mut VAL: u8 = 0;
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);
    let thread1 = thread::spawn(move || {
        let mut buf: [u8; 1] = [0; 1];
        // write() from the main thread will occur before the read() here
        // because preemption is disabled and the main thread yields after write().
        let res: i32 = unsafe {
            libc_utils::read_all(fds[1], buf.as_mut_ptr().cast(), buf.len() as libc::size_t)
                .try_into()
                .unwrap()
        };
        assert_eq!(res, 1);
        assert_eq!(buf, "a".as_bytes());
        // The read above establishes a happens-before so it is now safe to access this global variable.
        unsafe { assert_eq!(VAL, 1) };
    });
    unsafe { VAL = 1 };
    let data = "a".as_bytes().as_ptr();
    let res = unsafe { libc_utils::write_all(fds[0], data as *const libc::c_void, 1) };
    assert_eq!(res, 1);
    thread::yield_now();
    thread1.join().unwrap();
}

// Test the behaviour of a socketpair getting blocked on read and subsequently unblocked.
fn test_blocking_read() {
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);
    let thread1 = thread::spawn(move || {
        // Let this thread block on read.
        let mut buf: [u8; 3] = [0; 3];
        let res = unsafe {
            libc_utils::read_all(fds[1], buf.as_mut_ptr().cast(), buf.len() as libc::size_t)
        };
        assert_eq!(res, 3);
        assert_eq!(&buf, "abc".as_bytes());
    });
    let thread2 = thread::spawn(move || {
        // Unblock thread1 by doing writing something.
        let data = "abc".as_bytes().as_ptr();
        let res = unsafe { libc_utils::write_all(fds[0], data as *const libc::c_void, 3) };
        assert_eq!(res, 3);
    });
    thread1.join().unwrap();
    thread2.join().unwrap();
}

// Test the behaviour of a socketpair getting blocked on write and subsequently unblocked.
fn test_blocking_write() {
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);
    let arr1: [u8; 212992] = [1; 212992];
    // Exhaust the space in the buffer so the subsequent write will block.
    let res =
        unsafe { libc_utils::write_all(fds[0], arr1.as_ptr() as *const libc::c_void, 212992) };
    assert_eq!(res, 212992);
    let thread1 = thread::spawn(move || {
        let data = "abc".as_bytes().as_ptr();
        // The write below will be blocked because the buffer is already full.
        let res = unsafe { libc_utils::write_all(fds[0], data as *const libc::c_void, 3) };
        assert_eq!(res, 3);
    });
    let thread2 = thread::spawn(move || {
        // Unblock thread1 by freeing up some space.
        let mut buf: [u8; 3] = [0; 3];
        let res = unsafe {
            libc_utils::read_all(fds[1], buf.as_mut_ptr().cast(), buf.len() as libc::size_t)
        };
        assert_eq!(res, 3);
        assert_eq!(buf, [1, 1, 1]);
    });
    thread1.join().unwrap();
    thread2.join().unwrap();
}

/// Basic test for socketpair fcntl's F_SETFL and F_GETFL flag.
fn test_socketpair_setfl_getfl() {
    // Initialise socketpair fds.
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Test if both sides have O_RDWR.
    let res = unsafe { libc::fcntl(fds[0], libc::F_GETFL) };
    assert_eq!(res, libc::O_RDWR);
    let res = unsafe { libc::fcntl(fds[1], libc::F_GETFL) };
    assert_eq!(res, libc::O_RDWR);

    // Add the O_NONBLOCK flag with F_SETFL.
    let res = unsafe { libc::fcntl(fds[0], libc::F_SETFL, libc::O_NONBLOCK) };
    assert_eq!(res, 0);

    // Test if the O_NONBLOCK flag is successfully added.
    let res = unsafe { libc::fcntl(fds[0], libc::F_GETFL) };
    assert_eq!(res, libc::O_RDWR | libc::O_NONBLOCK);

    // The other side remains unchanged.
    let res = unsafe { libc::fcntl(fds[1], libc::F_GETFL) };
    assert_eq!(res, libc::O_RDWR);

    // Test if O_NONBLOCK flag can be unset.
    let res = unsafe { libc::fcntl(fds[0], libc::F_SETFL, 0) };
    assert_eq!(res, 0);
    let res = unsafe { libc::fcntl(fds[0], libc::F_GETFL) };
    assert_eq!(res, libc::O_RDWR);
}
