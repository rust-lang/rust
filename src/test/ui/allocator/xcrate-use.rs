// run-pass

// aux-build:custom.rs
// aux-build:helper.rs
// no-prefer-dynamic

#![feature(allocator_api)]

extern crate custom;
extern crate helper;

use std::alloc::{AllocInit, AllocRef, Global, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

#[global_allocator]
static GLOBAL: custom::A = custom::A(AtomicUsize::new(0));

fn main() {
    unsafe {
        let n = GLOBAL.0.load(Ordering::SeqCst);
        let layout = Layout::from_size_align(4, 2).unwrap();

        let memory = Global.alloc(layout.clone(), AllocInit::Uninitialized).unwrap();
        helper::work_with(&memory.ptr);
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 1);
        Global.dealloc(memory.ptr, layout);
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 2);

        let memory = System.alloc(layout.clone(), AllocInit::Uninitialized).unwrap();
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 2);
        helper::work_with(&memory.ptr);
        System.dealloc(memory.ptr, layout);
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 2);
    }
}
