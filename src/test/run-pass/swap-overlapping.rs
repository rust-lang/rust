// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #5041 - avoid overlapping memcpy when src and dest of a swap are the same

use std::ptr;
use std::util;

pub fn main() {
    let mut test = TestDescAndFn {
        desc: TestDesc {
            name: DynTestName(~"test"),
            should_fail: false
        },
        testfn: DynTestFn(|| ()),
    };
    do_swap(&mut test);
}

fn do_swap(test: &mut TestDescAndFn) {
    unsafe {
        util::swap_ptr(ptr::to_mut_unsafe_ptr(test),
                       ptr::to_mut_unsafe_ptr(test));
    }
}

pub enum TestName {
    DynTestName(~str)
}

pub enum TestFn {
    DynTestFn(~fn()),
    DynBenchFn(~fn(&mut int))
}

pub struct TestDesc {
    name: TestName,
    should_fail: bool
}

pub struct TestDescAndFn {
    desc: TestDesc,
    testfn: TestFn,
}
