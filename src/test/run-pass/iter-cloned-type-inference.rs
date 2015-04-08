// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test to see that the element type of .cloned() can be inferred
// properly. Previously this would fail to deduce the type of `sum`.


#![feature(core)]

fn square_sum(v: &[i64]) -> i64 {
    let sum: i64 = v.iter().cloned().sum();
    sum * sum
}

fn main() {
    assert_eq!(36, square_sum(&[1,2,3]));
}
