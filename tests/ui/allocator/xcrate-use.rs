//@ run-pass

//@ aux-build:custom.rs
//@ aux-build:helper.rs
//@ no-prefer-dynamic

#![feature(allocator_api)]

extern crate custom;
extern crate helper;

use std::alloc::{Alloc, Allocator, Global, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

#[global_allocator]
static GLOBAL: custom::A = custom::A(AtomicUsize::new(0));

fn main() {
    unsafe {
        let n = GLOBAL.0.load(Ordering::SeqCst);
        let layout = Layout::from_size_align(4, 2).unwrap();

        let memory = Global.alloc_ref().allocate(layout.clone()).unwrap();
        helper::work_with(&memory);
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 1);
        Global.alloc_ref().deallocate(memory, layout);
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 2);

        let memory = System.alloc_ref().allocate(layout.clone()).unwrap();
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 2);
        helper::work_with(&memory);
        System.alloc_ref().deallocate(memory, layout);
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 2);
    }
}
