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

static CONSTEXPR: *int = &'static 413 as *_;

pub fn main() {
    use std::vec::Vec;

    let x: Vec<_> = range(0u, 5).collect();
    assert_eq!(x.as_slice(), &[0u,1,2,3,4]);

    let x = range(0u, 5).collect::<Vec<_>>();
    assert_eq!(x.as_slice(), &[0u,1,2,3,4]);

    let y: _ = "hello";
    assert_eq!(y.len(), 5);

    let ptr = &5u;
    let ptr2 = ptr as *_;

    assert_eq!(ptr as *uint as uint, ptr2 as uint);
}
