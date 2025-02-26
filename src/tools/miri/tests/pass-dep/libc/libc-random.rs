//@ignore-target: windows # no libc
//@revisions: isolation no_isolation
//@[no_isolation]compile-flags: -Zmiri-disable-isolation

fn main() {
    test_getentropy();
    #[cfg(not(target_os = "macos"))]
    test_getrandom();
    #[cfg(any(target_os = "freebsd", target_os = "illumos", target_os = "solaris"))]
    test_arc4random_buf();
}

fn test_getentropy() {
    use libc::getentropy;

    let mut buf1 = [0u8; 256];
    let mut buf2 = [0u8; 257];
    unsafe {
        assert_eq!(getentropy(buf1.as_mut_ptr() as *mut libc::c_void, buf1.len()), 0);
        assert_eq!(getentropy(buf2.as_mut_ptr() as *mut libc::c_void, buf2.len()), -1);
        assert_eq!(std::io::Error::last_os_error().raw_os_error().unwrap(), libc::EIO);
    }
}

#[cfg(not(target_os = "macos"))]
fn test_getrandom() {
    use std::ptr;

    let mut buf = [0u8; 5];
    unsafe {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            assert_eq!(
                libc::syscall(
                    libc::SYS_getrandom,
                    ptr::null_mut::<libc::c_void>(),
                    0 as libc::size_t,
                    0 as libc::c_uint,
                ),
                0,
            );
            assert_eq!(
                libc::syscall(
                    libc::SYS_getrandom,
                    buf.as_mut_ptr() as *mut libc::c_void,
                    5 as libc::size_t,
                    0 as libc::c_uint,
                ),
                5,
            );
        }

        assert_eq!(
            libc::getrandom(ptr::null_mut::<libc::c_void>(), 0 as libc::size_t, 0 as libc::c_uint),
            0,
        );
        assert_eq!(
            libc::getrandom(
                buf.as_mut_ptr() as *mut libc::c_void,
                5 as libc::size_t,
                0 as libc::c_uint,
            ),
            5,
        );
    }
}

#[cfg(any(target_os = "freebsd", target_os = "illumos", target_os = "solaris"))]
fn test_arc4random_buf() {
    // FIXME: Use declaration from libc once <https://github.com/rust-lang/libc/pull/3944> lands.
    extern "C" {
        fn arc4random_buf(buf: *mut libc::c_void, size: libc::size_t);
    }
    let mut buf = [0u8; 5];
    unsafe { arc4random_buf(buf.as_mut_ptr() as _, buf.len()) };
    assert!(buf != [0u8; 5]);
}
