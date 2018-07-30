// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(allocator_api, nonnull)]

use std::alloc::{Alloc, Global, Layout, handle_alloc_error};

fn main() {
    unsafe {
        let ptr = Global.alloc_one::<i32>().unwrap_or_else(|_| {
            handle_alloc_error(Layout::new::<i32>())
        });
        *ptr.as_ptr() = 4;
        assert_eq!(*ptr.as_ptr(), 4);
        Global.dealloc_one(ptr);
    }
}
