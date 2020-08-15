// ignore-windows: No libc on Windows
// compile-flags: -Zmiri-disable-isolation

// FIXME: standard handles cannot be closed (https://github.com/rust-lang/rust/issues/40032)

#![feature(rustc_private)]

extern crate libc;

fn main() {
    unsafe {
        libc::close(1); //~ ERROR stdout cannot be closed
    }
}
