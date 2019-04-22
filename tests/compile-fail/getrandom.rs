// ignore-macos: Uses Linux-only APIs
// ignore-windows: Uses Linux-only APIs

#![feature(rustc_private)]
extern crate libc;

fn main() {
    let mut buf = [0u8; 5];
    unsafe {
        libc::syscall(libc::SYS_getrandom, buf.as_mut_ptr() as *mut libc::c_void, 5 as libc::size_t, 0 as libc::c_uint);
        //~^ ERROR miri does not support gathering system entropy in deterministic mode!
    }
}
