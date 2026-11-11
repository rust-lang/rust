//@ run-pass
// Testing guarantees provided by once functions.

#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

use std::sync::Arc;

fn foo<F: FnOnce()>(blk: F) {
    blk();
}

pub fn main() {
    let x = Arc::new(true);
    foo(use || {
        assert!(*x);
        drop(x);
    });
}
