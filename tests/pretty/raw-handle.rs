//@ pretty-mode:expanded
//@ pp-exact:raw-handle.pp
//@ edition: 2024

#![feature(field_projections)]
#![expect(incomplete_features)]

use std::ops::place::raw_handle;

struct S {
    a: i32,
}

fn main() {
    let s = S { a: 1 };
    let _ = unsafe { raw_handle!(s.a) };
}
