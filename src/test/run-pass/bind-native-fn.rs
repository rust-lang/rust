// From #1174:
// xfail-fast

use std;
import str;
import libc::*;

#[nolink]
native mod libc {
    fn write(fd: core::libc::c_int, buf: *u8, nbyte: core::libc::size_t);
}

fn main() {
    let s = "hello world\n";
    let b = str::bytes(s);
    let l = str::len(s) as core::libc::size_t;
    let b8 = unsafe { vec::unsafe::to_ptr(b) };
    libc::write(0i32, b8, l);
    let a = bind libc::write(0i32, _, _);
    a(b8, l);
}
