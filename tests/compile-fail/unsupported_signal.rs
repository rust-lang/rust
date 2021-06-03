//! `signal()` is special on Linux and macOS that it's only supported within libstd.
// ignore-windows: No libc on Windows
#![feature(rustc_private)]

extern crate libc;

fn main() {
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_IGN);
        //~^ ERROR unsupported operation: can't call foreign function: signal
    }
}
