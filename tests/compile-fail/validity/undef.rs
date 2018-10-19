#![allow(unused_variables)]
// error-pattern: encountered undefined address in pointer

use std::mem;

fn make_raw() -> *const f32 {
    unsafe { mem::uninitialized() }
}

fn main() {
    let _x = make_raw();
}
