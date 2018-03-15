// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(allocator_api, unique)]

use std::heap::{Heap, Alloc};

fn main() {
    unsafe {
        let ptr = Heap.alloc_one::<i32>().unwrap_or_else(|e| {
            Heap.oom(e)
        });
        *ptr.as_ptr() = 4;
        assert_eq!(*ptr.as_ptr(), 4);
        Heap.dealloc_one(ptr);
    }
}
