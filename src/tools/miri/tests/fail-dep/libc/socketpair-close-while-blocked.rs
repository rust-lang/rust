//! This is a regression test for <https://github.com/rust-lang/miri/issues/3947>: we had some
//! faulty logic around `release_clock` that led to this code not reporting a data race.
//~^^ERROR: deadlock
//@ignore-target: windows # no libc socketpair on Windows
//@compile-flags: -Zmiri-deterministic-concurrency
//@error-in-other-file: deadlock
use std::thread;

fn main() {
    let mut fds = [-1, -1];
    let res = unsafe { libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    let thread1 = thread::spawn(move || {
        let mut buf: [u8; 1] = [0; 1];
        let _res: i32 = unsafe {
            libc::read(fds[1], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) //~ERROR: deadlock
                .try_into()
                .unwrap()
        };
    });
    let thread2 = thread::spawn(move || {
        // Close the FD that the other thread is blocked on.
        unsafe { libc::close(fds[1]) };
    });

    // Run the other threads.
    thread::yield_now();

    // When they are both done, continue here.
    let data = "a".as_bytes().as_ptr();
    let res = unsafe { libc::write(fds[0], data as *const libc::c_void, 1) };
    assert_eq!(res, -1);

    thread1.join().unwrap();
    thread2.join().unwrap();
}
