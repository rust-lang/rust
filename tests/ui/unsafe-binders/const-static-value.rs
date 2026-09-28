//@ compile-flags: -Znext-solver
//@ check-pass

// Constants and statics can hold unsafe binders, and const-eval validates the
// value inside a binder like any other value. See #153362.

#![feature(unsafe_binders)]
#![allow(incomplete_features, dead_code)]

use std::unsafe_binder::wrap_binder;

const C: unsafe<> u8 = unsafe { wrap_binder!(1u8) };
static S: unsafe<'a> &'a u8 = unsafe { wrap_binder!(&1u8) };

fn main() {
    let _x = C;
}
