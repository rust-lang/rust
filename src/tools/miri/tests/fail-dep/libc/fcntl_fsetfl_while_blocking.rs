//@ignore-target: windows # Sockets/pipes are not implemented yet
//~^ ERROR: deadlock: the evaluated program deadlocked
//@compile-flags: -Zmiri-deterministic-concurrency
use std::thread;

/// If an O_NONBLOCK flag is set while the fd is blocking, that fd will not be woken up.
fn main() {
    let mut fds = [-1, -1];
    let res = unsafe { libc::pipe(fds.as_mut_ptr()) };
    assert_eq!(res, 0);
    let mut buf: [u8; 5] = [0; 5];
    let _thread1 = thread::spawn(move || {
        // Add O_NONBLOCK flag while pipe is still block on read.
        let res = unsafe { libc::fcntl(fds[0], libc::F_SETFL, libc::O_NONBLOCK) };
        assert_eq!(res, 0);
    });
    // Main thread will block on read.
    let _res = unsafe { libc::read(fds[0], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) };
    //~^ ERROR: deadlock: the evaluated program deadlocked
}
