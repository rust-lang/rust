//! This is a regression test for <https://github.com/rust-lang/miri/issues/3947>: we had some
//! faulty logic around `release_clock` that led to this code not reporting a data race.
//@ignore-target: windows # no libc socketpair on Windows
//@compile-flags: -Zmiri-deterministic-concurrency
use std::thread;

fn main() {
    static mut VAL: u8 = 0;
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);
    let thread1 = thread::spawn(move || {
        let data = "a".as_bytes().as_ptr();
        let res = unsafe { libc::write(fds[0], data as *const libc::c_void, 1) };
        assert_eq!(res, 1);
        // The write to VAL is *after* the write to the socket, so there's no proper synchronization.
        unsafe { VAL = 1 };
    });
    thread::yield_now();

    let mut buf: [u8; 1] = [0; 1];
    let res: i32 = unsafe {
        libc::read(fds[1], buf.as_mut_ptr().cast(), buf.len() as libc::size_t).try_into().unwrap()
    };
    assert_eq!(res, 1);
    assert_eq!(buf, "a".as_bytes());

    unsafe { assert_eq!({ VAL }, 1) }; //~ERROR: Data race

    thread1.join().unwrap();
}
