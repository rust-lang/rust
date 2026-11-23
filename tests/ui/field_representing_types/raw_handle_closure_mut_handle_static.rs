//@ check-pass
//@ known-bug: #145383
//@ edition: 2024

#![expect(incomplete_features)]
#![feature(field_projections)]

use std::ops::place::{MutHandle, raw_handle};

struct S {
    a: i32,
}

fn make_static() -> MutHandle<'static, i32> {
    let mut x = S { a: 1 };
    let r = &mut x;
    let c = || unsafe { raw_handle!(r.a) };
    c()
}

fn main() {
    let _h = make_static();
}
