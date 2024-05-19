//@ run-pass
#![allow(unused_variables)]
#![allow(stable_features)]

// test that ordinary fat pointer operations work.

#![feature(braced_empty_structs)]
#![feature(rustc_attrs)]

use std::sync::atomic;
use std::sync::atomic::Ordering::SeqCst;

static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(0);

struct DropMe {
}

impl Drop for DropMe {
    fn drop(&mut self) {
        COUNTER.fetch_add(1, SeqCst);
    }
}

fn fat_ptr_move_then_drop(a: Box<[DropMe]>) {
    let b = a;
}

fn main() {
    let a: Box<[DropMe]> = Box::new([DropMe { }]);
    fat_ptr_move_then_drop(a);
    assert_eq!(COUNTER.load(SeqCst), 1);
}
