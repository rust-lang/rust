//@ignore-target: windows # No libc pipe on Windows
// test_race depends on a deterministic schedule.
//@compile-flags: -Zmiri-deterministic-concurrency
use std::thread;

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::*;

fn main() {
    test_pipe();
    test_pipe_threaded();
    test_race();
    test_pipe_array();
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
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
    errno_check(unsafe { libc::pipe(fds.as_mut_ptr()) });

    // Read size == data available in buffer.
    let data = b"12345";
    write_all_from_slice(fds[1], data).unwrap();
    let buf3 = read_all_into_array::<5>(fds[0]).unwrap();
    assert_eq!(&buf3, data);

    // Read size > data available in buffer.
    let data = b"123";
    write_all_from_slice(fds[1], data).unwrap();
    let mut buf4: [u8; 5] = [0; 5];
    let (part1, rest) = read_into_slice(fds[0], &mut buf4).unwrap();
    assert_eq!(part1[..], data[..part1.len()]);
    // Write 2 more bytes so we can exactly fill the `rest`.
    write_all_from_slice(fds[1], b"34").unwrap();
    read_all_into_slice(fds[0], rest).unwrap();
}

fn test_pipe_threaded() {
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::pipe(fds.as_mut_ptr()) });

    let thread1 = thread::spawn(move || {
        let buf = read_all_into_array::<5>(fds[0]).unwrap();
        assert_eq!(&buf, b"abcde");
    });
    thread::yield_now();
    write_all_from_slice(fds[1], b"abcde").unwrap();
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

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#[allow(static_mut_refs)]
fn test_race() {
    static mut VAL: u8 = 0;
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::pipe(fds.as_mut_ptr()) });
    let thread1 = thread::spawn(move || {
        // write() from the main thread will occur before the read() here
        // because preemption is disabled and the main thread yields after write().
        let buf = read_all_into_array::<1>(fds[0]).unwrap();
        assert_eq!(&buf, b"a");
        // The read above establishes a happens-before so it is now safe to access this global variable.
        unsafe { assert_eq!(VAL, 1) };
    });
    unsafe { VAL = 1 };
    write_all_from_slice(fds[1], b"a").unwrap();
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
    target_os = "android",
    target_os = "illumos",
    target_os = "freebsd",
    target_os = "solaris"
))]
fn test_pipe2() {
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_NONBLOCK) });
}

/// Basic test for pipe fcntl's F_SETFL and F_GETFL flag.
fn test_pipe_setfl_getfl() {
    // Initialise pipe fds.
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::pipe(fds.as_mut_ptr()) });

    // Both sides should either have O_RONLY or O_WRONLY.
    assert_eq!(
        errno_result(unsafe { libc::fcntl(fds[0], libc::F_GETFL) }).unwrap(),
        libc::O_RDONLY
    );
    assert_eq!(
        errno_result(unsafe { libc::fcntl(fds[1], libc::F_GETFL) }).unwrap(),
        libc::O_WRONLY
    );

    // Add the O_NONBLOCK flag with F_SETFL.
    errno_check(unsafe { libc::fcntl(fds[0], libc::F_SETFL, libc::O_NONBLOCK) });

    // Test if the O_NONBLOCK flag is successfully added.
    assert_eq!(
        errno_result(unsafe { libc::fcntl(fds[0], libc::F_GETFL) }).unwrap(),
        libc::O_RDONLY | libc::O_NONBLOCK
    );

    // The other side remains unchanged.
    assert_eq!(
        errno_result(unsafe { libc::fcntl(fds[1], libc::F_GETFL) }).unwrap(),
        libc::O_WRONLY
    );

    // Test if O_NONBLOCK flag can be unset.
    errno_check(unsafe { libc::fcntl(fds[0], libc::F_SETFL, 0) });
    assert_eq!(
        errno_result(unsafe { libc::fcntl(fds[0], libc::F_GETFL) }).unwrap(),
        libc::O_RDONLY
    );
}

/// Test the behaviour of F_SETFL/F_GETFL when a fd is blocking.
/// The expected execution is:
/// 1. Main thread blocks on fds[0] `read`.
/// 2. Thread 1 sets O_NONBLOCK flag on fds[0],
///    checks the value of F_GETFL,
///    then writes to fds[1] to unblock main thread's `read`.
fn test_pipe_fcntl_threaded() {
    let mut fds = [-1, -1];
    errno_check(unsafe { libc::pipe(fds.as_mut_ptr()) });
    let thread1 = thread::spawn(move || {
        // Add O_NONBLOCK flag while pipe is still blocked on read.
        errno_check(unsafe { libc::fcntl(fds[0], libc::F_SETFL, libc::O_NONBLOCK) });

        // Check the new flag value while the main thread is still blocked on fds[0].
        assert_eq!(
            errno_result(unsafe { libc::fcntl(fds[0], libc::F_GETFL) }).unwrap(),
            libc::O_NONBLOCK
        );

        // The write below will unblock the `read` in main thread: even though
        // the socket is now "non-blocking", the shim needs to deal correctly
        // with threads that were blocked before the socket was made non-blocking.
        write_all_from_slice(fds[1], b"abcde").unwrap();
    });
    // The `read` below will block.
    let buf = read_all_into_array::<5>(fds[0]).unwrap();
    thread1.join().unwrap();
    assert_eq!(&buf, b"abcde");
}
