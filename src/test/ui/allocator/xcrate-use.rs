// run-pass

// aux-build:custom.rs
// aux-build:helper.rs
// no-prefer-dynamic

#![feature(allocator_api)]
#![feature(slice_ptr_get)]

extern crate custom;
extern crate helper;

use std::alloc::{Allocator, Global, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

#[global_allocator]
static GLOBAL: custom::A = custom::A(AtomicUsize::new(0));

fn main() {
    unsafe {
        let n = GLOBAL.0.load(Ordering::SeqCst);
        let layout = Layout::from_size_align(4, 2).unwrap();

        let memory = Global.allocate(layout.clone()).unwrap();
        helper::work_with(&memory);
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 1);
        Global.deallocate(memory.as_non_null_ptr(), layout);
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 2);

        let memory = System.allocate(layout.clone()).unwrap();
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 2);
        helper::work_with(&memory);
        System.deallocate(memory.as_non_null_ptr(), layout);
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 2);
    }
}
