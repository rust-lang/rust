//@ignore-target: windows # No libc errno on Windows

/// Tests whether each thread has its own `__errno_location`.
fn main() {
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
