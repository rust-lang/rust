//@ignore-target: windows # No libc pipe on Windows
// test_race depends on a deterministic schedule.
//@compile-flags: -Zmiri-preemption-rate=0
use std::thread;
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
}

fn test_pipe() {
    let mut fds = [-1, -1];
    let res = unsafe { libc::pipe(fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    // Read size == data available in buffer.
    let data = "12345".as_bytes().as_ptr();
    let res = unsafe { libc::write(fds[1], data as *const libc::c_void, 5) };
    assert_eq!(res, 5);
    let mut buf3: [u8; 5] = [0; 5];
    let res = unsafe { libc::read(fds[0], buf3.as_mut_ptr().cast(), buf3.len() as libc::size_t) };
    assert_eq!(res, 5);
    assert_eq!(buf3, "12345".as_bytes());

    // Read size > data available in buffer.
    let data = "123".as_bytes().as_ptr();
    let res = unsafe { libc::write(fds[1], data as *const libc::c_void, 3) };
    assert_eq!(res, 3);
    let mut buf4: [u8; 5] = [0; 5];
    let res = unsafe { libc::read(fds[0], buf4.as_mut_ptr().cast(), buf4.len() as libc::size_t) };
    assert_eq!(res, 3);
    assert_eq!(&buf4[0..3], "123".as_bytes());
}

fn test_pipe_threaded() {
    let mut fds = [-1, -1];
    let res = unsafe { libc::pipe(fds.as_mut_ptr()) };
    assert_eq!(res, 0);

    let thread1 = thread::spawn(move || {
        let mut buf: [u8; 5] = [0; 5];
        let res: i64 = unsafe {
            libc::read(fds[0], buf.as_mut_ptr().cast(), buf.len() as libc::size_t)
                .try_into()
                .unwrap()
        };
        assert_eq!(res, 5);
        assert_eq!(buf, "abcde".as_bytes());
    });
    // FIXME: we should yield here once blocking is implemented.
    //thread::yield_now();
    let data = "abcde".as_bytes().as_ptr();
    let res = unsafe { libc::write(fds[1], data as *const libc::c_void, 5) };
    assert_eq!(res, 5);
    thread1.join().unwrap();

    // Read and write from different direction
    let thread2 = thread::spawn(move || {
        // FIXME: we should yield here once blocking is implemented.
        //thread::yield_now();
        let data = "12345".as_bytes().as_ptr();
        let res = unsafe { libc::write(fds[1], data as *const libc::c_void, 5) };
        assert_eq!(res, 5);
    });
    // FIXME: we should not yield here once blocking is implemented.
    thread::yield_now();
    let mut buf: [u8; 5] = [0; 5];
    let res = unsafe { libc::read(fds[0], buf.as_mut_ptr().cast(), buf.len() as libc::size_t) };
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
            libc::read(fds[0], buf.as_mut_ptr().cast(), buf.len() as libc::size_t)
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
    let res = unsafe { libc::write(fds[1], data as *const libc::c_void, 1) };
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
