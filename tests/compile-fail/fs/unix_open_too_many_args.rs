// ignore-windows: No libc on Windows
// compile-flags: -Zmiri-disable-isolation

#![feature(rustc_private)]

extern crate libc;

fn main() {
    test_open_too_many_args();
}

fn test_open_too_many_args() {
    let name = b"too_many_args.txt\0";
    let name_ptr = name.as_ptr().cast::<libc::c_char>();
    let _fd = unsafe { libc::open(name_ptr, libc::O_RDONLY, 0, 0) }; //~ ERROR Undefined Behavior: incorrect number of arguments for `open`: got 4, expected 2 or 3
}
