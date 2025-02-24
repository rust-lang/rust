//@ignore-target: windows # only very limited libc on Windows
//@compile-flags: -Zmiri-disable-isolation
#![feature(io_error_more)]
#![feature(pointer_is_aligned_to)]

use std::mem::transmute;

/// Tests whether each thread has its own `__errno_location`.
fn test_thread_local_errno() {
    #[cfg(any(target_os = "illumos", target_os = "solaris"))]
    use libc::___errno as __errno_location;
    #[cfg(target_os = "android")]
    use libc::__errno as __errno_location;
    #[cfg(target_os = "linux")]
    use libc::__errno_location;
    #[cfg(any(target_os = "freebsd", target_os = "macos"))]
    use libc::__error as __errno_location;

    unsafe {
        *__errno_location() = 0xBEEF;
        std::thread::spawn(|| {
            assert_eq!(*__errno_location(), 0);
            *__errno_location() = 0xBAD1DEA;
            assert_eq!(*__errno_location(), 0xBAD1DEA);
        })
        .join()
        .unwrap();
        assert_eq!(*__errno_location(), 0xBEEF);
    }
}

fn test_environ() {
    // Just a smoke test for now, checking that the extern static exists.
    extern "C" {
        static mut environ: *const *const libc::c_char;
    }

    unsafe {
        let mut e = environ;
        // Iterate to the end.
        while !(*e).is_null() {
            e = e.add(1);
        }
    }
}

#[cfg(target_os = "linux")]
fn test_sigrt() {
    let min = libc::SIGRTMIN();
    let max = libc::SIGRTMAX();

    // "The Linux kernel supports a range of 33 different real-time
    // signals, numbered 32 to 64"
    assert!(min >= 32);
    assert!(max >= 32);
    assert!(min <= 64);
    assert!(max <= 64);

    // "POSIX.1-2001 requires that an implementation support at least
    // _POSIX_RTSIG_MAX (8) real-time signals."
    assert!(min < max);
    assert!(max - min >= 8)
}

fn test_dlsym() {
    let addr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, b"notasymbol\0".as_ptr().cast()) };
    assert!(addr as usize == 0);

    let addr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, b"isatty\0".as_ptr().cast()) };
    assert!(addr as usize != 0);
    let isatty: extern "C" fn(i32) -> i32 = unsafe { transmute(addr) };
    assert_eq!(isatty(999), 0);
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap();
    assert_eq!(errno, libc::EBADF);
}

fn test_getuid() {
    let _val = unsafe { libc::getuid() };
}

fn test_geteuid() {
    let _val = unsafe { libc::geteuid() };
}

fn main() {
    test_thread_local_errno();
    test_environ();
    test_dlsym();
    test_getuid();
    test_geteuid();

    #[cfg(target_os = "linux")]
    test_sigrt();
}
