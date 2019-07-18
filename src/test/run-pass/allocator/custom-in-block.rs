// run-pass
// no-prefer-dynamic
// aux-build:custom.rs
// aux-build:helper.rs

extern crate custom;
extern crate helper;

use custom::A;
use std::sync::atomic::{AtomicUsize, Ordering};

fn main() {
    #[global_allocator]
    pub static GLOBAL: A = A(AtomicUsize::new(0));

    let s = Box::new(0);
    helper::work_with(&s);
    assert_eq!(GLOBAL.0.load(Ordering::SeqCst), 1);
    drop(s);
    assert_eq!(GLOBAL.0.load(Ordering::SeqCst), 2);
}
