// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:helper.rs
// no-prefer-dynamic

#![feature(global_allocator, heap_api, allocator_api)]

extern crate helper;

use std::heap::{Heap, Alloc, System, Layout, AllocErr};
use std::sync::atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT};

static HITS: AtomicUsize = ATOMIC_USIZE_INIT;

struct A;

unsafe impl<'a> Alloc for &'a A {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
        HITS.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }

    unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
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

        let ptr = Heap.alloc(layout.clone()).unwrap();
        helper::work_with(&ptr);
        assert_eq!(HITS.load(Ordering::SeqCst), n + 1);
        Heap.dealloc(ptr, layout.clone());
        assert_eq!(HITS.load(Ordering::SeqCst), n + 2);

        let s = String::with_capacity(10);
        helper::work_with(&s);
        assert_eq!(HITS.load(Ordering::SeqCst), n + 3);
        drop(s);
        assert_eq!(HITS.load(Ordering::SeqCst), n + 4);

        let ptr = System.alloc(layout.clone()).unwrap();
        assert_eq!(HITS.load(Ordering::SeqCst), n + 4);
        helper::work_with(&ptr);
        System.dealloc(ptr, layout);
        assert_eq!(HITS.load(Ordering::SeqCst), n + 4);
    }
}
