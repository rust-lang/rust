// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:out of stack while using stackalloc

#![feature(core, test)]

use std::intrinsics::{
    stacksave,
    stackalloc,
    stackrestore,
    size_of,
};

extern crate test;

fn main() {
    let n = std::env::args().count() + 1000000;
    for _ in 1..100 {
        unsafe {
            let n = size_of::<i32>()*n;
            let data1: *mut i32 = stackalloc(n);
            let data2 = std::raw::Slice {
                data: data1 as *const i32,
                len: n,
            };
            let data3: &mut [i32] = std::mem::transmute(data2);
            data3[n - 1] = 42;
            test::black_box(data3);
        }
    }
}

#[no_mangle]
pub extern fn __morestack_allocate_stack_space() {
    panic!("out of stack while using stackalloc");
}
