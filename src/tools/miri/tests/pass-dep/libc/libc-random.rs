//@ignore-target: windows # no libc
//@revisions: isolation no_isolation
//@[no_isolation]compile-flags: -Zmiri-disable-isolation

fn main() {
    test_getentropy();
    #[cfg(not(target_os = "macos"))]
    test_getrandom();
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
        #[cfg(target_os = "linux")]
        assert_eq!(
            libc::syscall(
                libc::SYS_getrandom,
                ptr::null_mut::<libc::c_void>(),
                0 as libc::size_t,
                0 as libc::c_uint,
            ),
            0,
        );
        #[cfg(target_os = "linux")]
        assert_eq!(
            libc::syscall(
                libc::SYS_getrandom,
                buf.as_mut_ptr() as *mut libc::c_void,
                5 as libc::size_t,
                0 as libc::c_uint,
            ),
            5,
        );

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
