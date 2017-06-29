// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(alloc, allocator_api, heap_api, unique)]

extern crate alloc;

use alloc::heap::HeapAlloc;
use alloc::allocator::Alloc;

fn main() {
    unsafe {
        let ptr = HeapAlloc.alloc_one::<i32>().unwrap_or_else(|e| {
            HeapAlloc.oom(e)
        });
        *ptr.as_ptr() = 4;
        assert_eq!(*ptr.as_ptr(), 4);
        HeapAlloc.dealloc_one(ptr);
    }
}
