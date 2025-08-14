//@ignore-target: windows # No libc pipe on Windows
// test_race depends on a deterministic schedule.
//@compile-flags: -Zmiri-deterministic-concurrency
use std::thread;

#[path = "../../utils/libc.rs"]
mod libc_utils;

fn main() {
    test_pipe();
    test_pipe_threaded();
    test_race();
    test_pipe_array();
    #[cfg(any(
        target_os = "linux",
        target_os = "illumos",
        target_os = "freebsd",
        target_os = "solaris"
    ))]
    // `pipe2` only exists in some specific os.
    test_pipe2();
    test_pipe_setfl_getfl();
    test_pipe_fcntl_threaded();
}

fn test_pipe() {
    let mut fds = [-1, -1];
    let res = unsafe { libc::pipe(fds.as_mut_ptr()) };
    assert_eq!(res, 0);

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
}

fn test_pipe_threaded() {
    let mut fds = [-1, -1];
    let res = unsafe { libc::pipe(fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    let thread1 = thread::spawn(move || {
        let mut buf: [u8; 5] = [0; 5];
        let res: i64 = unsafe {
            libc_utils::read_all(fds[0], buf.as_mut_ptr().cast(), buf.len() as libc::size_t)
                .try_into()
                .unwrap()
        };
        assert_eq!(res, 5);
        assert_eq!(buf, "abcde".as_bytes());
    });
    thread::yield_now();
    let data = "abcde".as_bytes().as_ptr();
    let res = unsafe { libc_utils::write_all(fds[1], data as *const libc::c_void, 5) };
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

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#[allow(static_mut_refs)]
fn test_race() {
    static mut VAL: u8 = 0;
    let mut fds = [-1, -1];
    let res = unsafe { libc::pipe(fds.as_mut_ptr()) };
    assert_eq!(res, 0);
    let thread1 = thread::spawn(move || {
        let mut buf: [u8; 1] = [0; 1];
        // write() from the main thread will occur before the read() here
        // because preemption is disabled and the main thread yields after write().
        let res: i32 = unsafe {
            libc_utils::read_all(fds[0], buf.as_mut_ptr().cast(), buf.len() as libc::size_t)
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
    let res = unsafe { libc_utils::write_all(fds[1], data as *const libc::c_void, 1) };
    assert_eq!(res, 1);
    thread::yield_now();
    thread1.join().unwrap();
}

fn test_pipe_array() {
    // Declare `pipe` to take an array rather than a `*mut i32`.
    extern "C" {
        fn pipe(pipefd: &mut [i32; 2]) -> i32;
    }

    let mut fds: [i32; 2] = [0; 2];
    assert_eq!(unsafe { pipe(&mut fds) }, 0);
}

/// Test if pipe2 (including the O_NONBLOCK flag) is supported.
#[cfg(any(
    target_os = "linux",
    target_os = "illumos",
    target_os = "freebsd",
    target_os = "solaris"
))]
fn test_pipe2() {
    let mut fds = [-1, -1];
    let res = unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_NONBLOCK) };
    assert_eq!(res, 0);
}

/// Basic test for pipe fcntl's F_SETFL and F_GETFL flag.
fn test_pipe_setfl_getfl() {
    // Initialise pipe fds.
    let mut fds = [-1, -1];
    let res = unsafe { libc::pipe(fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Both sides should either have O_RONLY or O_WRONLY.
    let res = unsafe { libc::fcntl(fds[0], libc::F_GETFL) };
    assert_eq!(res, libc::O_RDONLY);
    let res = unsafe { libc::fcntl(fds[1], libc::F_GETFL) };
    assert_eq!(res, libc::O_WRONLY);

    // Add the O_NONBLOCK flag with F_SETFL.
    let res = unsafe { libc::fcntl(fds[0], libc::F_SETFL, libc::O_NONBLOCK) };
    assert_eq!(res, 0);

    // Test if the O_NONBLOCK flag is successfully added.
    let res = unsafe { libc::fcntl(fds[0], libc::F_GETFL) };
    assert_eq!(res, libc::O_RDONLY | libc::O_NONBLOCK);

    // The other side remains unchanged.
    let res = unsafe { libc::fcntl(fds[1], libc::F_GETFL) };
    assert_eq!(res, libc::O_WRONLY);

    // Test if O_NONBLOCK flag can be unset.
    let res = unsafe { libc::fcntl(fds[0], libc::F_SETFL, 0) };
    assert_eq!(res, 0);
    let res = unsafe { libc::fcntl(fds[0], libc::F_GETFL) };
    assert_eq!(res, libc::O_RDONLY);
}

/// Test the behaviour of F_SETFL/F_GETFL when a fd is blocking.
/// The expected execution is:
/// 1. Main thread blocks on fds[0] `read`.
/// 2. Thread 1 sets O_NONBLOCK flag on fds[0],
///    checks the value of F_GETFL,
///    then writes to fds[1] to unblock main thread's `read`.
fn test_pipe_fcntl_threaded() {
    let mut fds = [-1, -1];
    let res = unsafe { libc::pipe(fds.as_mut_ptr()) };
    assert_eq!(res, 0);
    let mut buf: [u8; 5] = [0; 5];
    let thread1 = thread::spawn(move || {
        // Add O_NONBLOCK flag while pipe is still blocked on read.
        let res = unsafe { libc::fcntl(fds[0], libc::F_SETFL, libc::O_NONBLOCK) };
        assert_eq!(res, 0);

        // Check the new flag value while the main thread is still blocked on fds[0].
        let res = unsafe { libc::fcntl(fds[0], libc::F_GETFL) };
        assert_eq!(res, libc::O_NONBLOCK);

        // The write below will unblock the `read` in main thread: even though
        // the socket is now "non-blocking", the shim needs to deal correctly
        // with threads that were blocked before the socket was made non-blocking.
        let data = "abcde".as_bytes().as_ptr();
        let res = unsafe { libc_utils::write_all(fds[1], data as *const libc::c_void, 5) };
        assert_eq!(res, 5);
    });
    // The `read` below will block.
    let res =
        unsafe { libc_utils::read_all(fds[0], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) };
    thread1.join().unwrap();
    assert_eq!(res, 5);
}
