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
// aux-build:custom-as-global.rs
// aux-build:helper.rs
// no-prefer-dynamic

#![feature(heap_api, allocator_api)]

extern crate custom;
extern crate custom_as_global;
extern crate helper;

use std::alloc::{alloc, dealloc, GlobalAlloc, System, Layout};
use std::sync::atomic::{Ordering, ATOMIC_USIZE_INIT};

static GLOBAL: custom::A = custom::A(ATOMIC_USIZE_INIT);

fn main() {
    unsafe {
        let n = custom_as_global::get();
        let layout = Layout::from_size_align(4, 2).unwrap();

        // Global allocator routes to the `custom_as_global` global
        let ptr = alloc(layout.clone());
        helper::work_with(&ptr);
        assert_eq!(custom_as_global::get(), n + 1);
        dealloc(ptr, layout.clone());
        assert_eq!(custom_as_global::get(), n + 2);

        // Usage of the system allocator avoids all globals
        let ptr = System.alloc(layout.clone());
        helper::work_with(&ptr);
        assert_eq!(custom_as_global::get(), n + 2);
        System.dealloc(ptr, layout.clone());
        assert_eq!(custom_as_global::get(), n + 2);

        // Usage of our personal allocator doesn't affect other instances
        let ptr = GLOBAL.alloc(layout.clone());
        helper::work_with(&ptr);
        assert_eq!(custom_as_global::get(), n + 2);
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), 1);
        GLOBAL.dealloc(ptr, layout);
        assert_eq!(custom_as_global::get(), n + 2);
        assert_eq!(GLOBAL.0.load(Ordering::SeqCst), 2);
    }
}

