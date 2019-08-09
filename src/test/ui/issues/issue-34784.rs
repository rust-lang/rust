// run-pass
#![allow(dead_code)]
const C: *const u8 = &0;

fn foo(x: *const u8) {
    match x {
        C => {}
        _ => {}
    }
}

const D: *const [u8; 4] = b"abcd";

fn main() {
    match D {
        D => {}
        _ => {}
    }
}
