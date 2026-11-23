#![feature(field_projections)]

use std::ops::place::raw_handle;

struct S {
    a: i32,
}

fn main() {
    let s = S { a: 1 };
    let _ = unsafe { raw_handle!(s.a) };
    let _ = unsafe { raw_handle!((*s).a) };
}
