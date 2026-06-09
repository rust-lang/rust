//@ check-pass

#![feature(unsafe_binders)]

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
