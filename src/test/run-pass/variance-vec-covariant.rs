// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that vec is now covariant in its argument type.

#![allow(dead_code)]

fn foo<'a,'b>(v1: Vec<&'a i32>, v2: Vec<&'b i32>) -> i32 {
    bar(v1, v2).cloned().unwrap_or(0) // only type checks if we can intersect 'a and 'b
}

fn bar<'c>(v1: Vec<&'c i32>, v2: Vec<&'c i32>) -> Option<&'c i32> {
    v1.get(0).cloned().or_else(|| v2.get(0).cloned())
}

fn main() {
    let x = 22;
    let y = 44;
    assert_eq!(foo(vec![&x], vec![&y]), 22);
    assert_eq!(foo(vec![&y], vec![&x]), 44);
}
