#![feature(rustc_private)]
extern crate libc;

fn main() {
    let mut buf = [0u8; 5];
    unsafe {
        libc::syscall(libc::SYS_getrandom, buf.as_mut_ptr() as *mut libc::c_void, 5, 0);
        //~^ ERROR constant evaluation error: miri does not support random number generators in deterministic mode!
    }
}
