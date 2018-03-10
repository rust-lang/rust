// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:custom.rs
// aux-build:helper.rs
// no-prefer-dynamic

#![feature(global_allocator, heap_api, allocator_api)]

extern crate custom;
extern crate helper;

use std::heap::{Heap, Alloc, System, Layout};
use std::sync::atomic::{Ordering, ATOMIC_USIZE_INIT};

#[global_allocator]
static GLOBAL: custom::A = custom::A(ATOMIC_USIZE_INIT);

fn main() {
    unsafe {
        let n = GLOBAL.0.load(Ordering::SeqCst);
        let layout = Layout::from_size_align(4, 2).unwrap();

        let ptr = Heap.alloc(layout.clone()).unwrap();
        helper::work_with(&ptr);
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 1);
        Heap.dealloc(ptr, layout.clone());
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 2);

        let ptr = System.alloc(layout.clone()).unwrap();
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 2);
        helper::work_with(&ptr);
        System.dealloc(ptr, layout);
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), n + 2);
    }
}
