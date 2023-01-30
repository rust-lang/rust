// run-pass
// Testing guarantees provided by once functions.



use std::sync::Arc;

fn foo<F:FnOnce()>(blk: F) {
    blk();
}

pub fn main() {
    let x = Arc::new(true);
    foo(move|| {
        assert!(*x);
        drop(x);
    });
}
