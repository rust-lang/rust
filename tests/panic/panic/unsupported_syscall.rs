//@ignore-windows: No libc on Windows
//@ignore-apple: `syscall` is not supported on macOS
//@compile-flags: -Zmiri-panic-on-unsupported
#![feature(rustc_private)]

extern crate libc;

fn main() {
    unsafe {
        libc::syscall(0);
    }
}
