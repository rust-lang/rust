// run-pass

// aux-build:helper.rs
// no-prefer-dynamic

#![feature(allocator_api)]

extern crate helper;

use std::alloc::{self, AllocInit, AllocRef, Global, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static HITS: AtomicUsize = AtomicUsize::new(0);

struct A;

unsafe impl alloc::GlobalAlloc for A {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        HITS.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        HITS.fetch_add(1, Ordering::SeqCst);
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: A = A;

fn main() {
    println!("hello!");

    let n = HITS.load(Ordering::SeqCst);
    assert!(n > 0);
    unsafe {
        let layout = Layout::from_size_align(4, 2).unwrap();

        let memory = Global.alloc(layout.clone(), AllocInit::Uninitialized).unwrap();
        helper::work_with(&memory.ptr);
        assert_eq!(HITS.load(Ordering::SeqCst), n + 1);
        Global.dealloc(memory.ptr, layout);
        assert_eq!(HITS.load(Ordering::SeqCst), n + 2);

        let s = String::with_capacity(10);
        helper::work_with(&s);
        assert_eq!(HITS.load(Ordering::SeqCst), n + 3);
        drop(s);
        assert_eq!(HITS.load(Ordering::SeqCst), n + 4);

        let memory = System.alloc(layout.clone(), AllocInit::Uninitialized).unwrap();
        assert_eq!(HITS.load(Ordering::SeqCst), n + 4);
        helper::work_with(&memory.ptr);
        System.dealloc(memory.ptr, layout);
        assert_eq!(HITS.load(Ordering::SeqCst), n + 4);
    }
}
