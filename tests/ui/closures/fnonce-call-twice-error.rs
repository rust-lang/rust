//! Test that `FnOnce` closures cannot be called twice.

use std::sync::Arc;

fn foo<F: FnOnce()>(blk: F) {
    blk();
    blk(); //~ ERROR use of moved value
}

fn main() {
    let x = Arc::new(true);
    foo(move || {
        assert!(*x);
        drop(x);
    });
}
