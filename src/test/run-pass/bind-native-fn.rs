// From #1174:
// xfail-test bots are crashing on this on x86_64

use std;
import str;
import ctypes::*;

#[link_name = ""]
native mod libc {
    fn write(fd: c_int, buf: *u8, nbyte: size_t);
}

fn main() {
    let s = "hello world\n";
    let b = str::bytes(s);
    let l = str::byte_len(s);
    let b8 = unsafe { vec::unsafe::to_ptr(b) };
    libc::write(0i32, b8, l);
    let a = bind libc::write(0i32, _, _);
    a(b8, l);
}
