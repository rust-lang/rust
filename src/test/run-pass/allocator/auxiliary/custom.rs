// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic

#![feature(heap_api, allocator_api)]
#![crate_type = "rlib"]

use std::heap::{Alloc, System, AllocErr, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct A(pub AtomicUsize);

unsafe impl<'a> Alloc for &'a A {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
        self.0.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }

    unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        self.0.fetch_add(1, Ordering::SeqCst);
        System.dealloc(ptr, layout)
    }
}
