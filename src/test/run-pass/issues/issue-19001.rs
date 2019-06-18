// run-pass
#![allow(dead_code)]
// check that we handle recursive arrays correctly in `type_of`

struct Loopy {
    ptr: *mut [Loopy; 1]
}

fn main() {
    let _t = Loopy { ptr: core::ptr::null_mut() };
}
