//@ run-pass
//! This tests `RefCell` is usable in `const` contexts

#![feature(const_ref_cell)]

use std::mem;
use std::sync::atomic::{AtomicU8, Ordering};
use std::cell::RefCell;

static DROP_COUNT: AtomicU8 = AtomicU8::new(0);

struct Dummy;

impl Drop for Dummy {
    fn drop(&mut self){
        DROP_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

const REF_CELL_TEST: RefCell<Dummy> = {
    let a = RefCell::new(Dummy);
    let b = RefCell::new(Dummy);

    // Check that `replace` is usable at compile-time
    mem::forget(a.replace(Dummy));

    // Check that `swap` is usable at compile-time
    a.swap(&b);

    // Forget `b` to avoid drop
    mem::forget(b);
    a
};

fn main() {
    let dummy = REF_CELL_TEST;
    assert_eq!(DROP_COUNT.load(Ordering::Relaxed), 0);
    mem::forget(dummy);
}
