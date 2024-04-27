//@ignore-target-windows: no libc
//@ignore-target-apple: no getrandom

use std::ptr;

fn main() {
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
