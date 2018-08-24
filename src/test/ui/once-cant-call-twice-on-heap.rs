// Testing guarantees provided by once functions.
// This program would segfault if it were legal.

#![feature(once_fns)]
use std::sync::Arc;

fn foo<F:FnOnce()>(blk: F) {
    blk();
    blk(); //~ ERROR use of moved value
}

fn main() {
    let x = Arc::new(true);
    foo(move|| {
        assert!(*x);
        drop(x);
    });
}
