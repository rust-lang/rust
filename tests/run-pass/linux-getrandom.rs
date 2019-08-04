// Unfortunately, compiletest_rs does not support 'only-linux',
// so we need to ignore Windows and macOS instead.
// ignore-macos: Uses Linux-only APIs
// ignore-windows: Uses Linux-only APIs
#![feature(rustc_private)]
extern crate libc;

fn main() {
    let mut buf = [0u8; 5];
    unsafe {
        assert_eq!(libc::syscall(libc::SYS_getrandom, 0 as *mut libc::c_void, 0 as libc::size_t, 0 as libc::c_uint), 0);
        assert_eq!(libc::syscall(libc::SYS_getrandom, buf.as_mut_ptr() as *mut libc::c_void, 5 as libc::size_t, 0 as libc::c_uint), 5);

        assert_eq!(libc::getrandom(0 as *mut libc::c_void, 0 as libc::size_t, 0 as libc::c_uint), 0);
        assert_eq!(libc::getrandom(buf.as_mut_ptr() as *mut libc::c_void, 5 as libc::size_t, 0 as libc::c_uint), 5);
    }
}
