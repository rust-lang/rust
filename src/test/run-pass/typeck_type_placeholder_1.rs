// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test checks that the `_` type placeholder works
// correctly for enabling type inference.

// pretty-expanded FIXME #23616

struct TestStruct {
    x: *const isize
}

unsafe impl Sync for TestStruct {}

static CONSTEXPR: TestStruct = TestStruct{ x: &413 };


pub fn main() {
    let x: Vec<_> = (0..5).collect();
    let expected: &[usize] = &[0,1,2,3,4];
    assert_eq!(x, expected);

    let x = (0..5).collect::<Vec<_>>();
    assert_eq!(x, expected);

    let y: _ = "hello";
    assert_eq!(y.len(), 5);

    let ptr: &usize = &5;
    let ptr2 = ptr as *const _;

    assert_eq!(ptr as *const usize as usize, ptr2 as usize);
}
