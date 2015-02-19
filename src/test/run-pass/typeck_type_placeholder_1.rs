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

struct TestStruct {
    x: *const int
}

unsafe impl Sync for TestStruct {}

static CONSTEXPR: TestStruct = TestStruct{x: &413 as *const _};


pub fn main() {
    let x: Vec<_> = (0_usize..5).collect();
    let expected: &[uint] = &[0,1,2,3,4];
    assert_eq!(x, expected);

    let x = (0_usize..5).collect::<Vec<_>>();
    assert_eq!(x, expected);

    let y: _ = "hello";
    assert_eq!(y.len(), 5);

    let ptr = &5_usize;
    let ptr2 = ptr as *const _;

    assert_eq!(ptr as *const uint as uint, ptr2 as uint);
}
