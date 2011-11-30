// From #1174:
// xfail-test bots are crashing on this on x86_64

use std;

#[link_name = ""]
native mod libc {
    fn printf(s: *u8, a: int); /* A tenuous definition. */
}

fn main() {
    let b = std::str::bytes("%d\n");
    let b8 = unsafe { std::vec::unsafe::to_ptr(b) };
    libc::printf(b8, 4);
    let a = bind libc::printf(b8, 5);
    a(); /* core dump */
}
