//@ check-pass

#![feature(unsafe_binders)]
//~^ WARN the feature `unsafe_binders` is incomplete

use std::unsafe_binder::wrap_binder;

struct A {
    b: unsafe<> (),
}

fn main() {
    unsafe {
        let _ = A {
            b: wrap_binder!(()),
        };
    }
}
