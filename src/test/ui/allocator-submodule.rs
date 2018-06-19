// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that it is possible to create a global allocator in a submodule, rather than in the crate
// root.

#![feature(alloc, allocator_api, global_allocator)]

extern crate alloc;

use std::{
    alloc::{GlobalAlloc, Layout},
    ptr,
};

struct MyAlloc;

unsafe impl GlobalAlloc for MyAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ptr::null_mut()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {}
}

mod submod {
    use super::MyAlloc;

    #[global_allocator]
    static MY_HEAP: MyAlloc = MyAlloc; //~ ERROR global_allocator
}

fn main() {}
