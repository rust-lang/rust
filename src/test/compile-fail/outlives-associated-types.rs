// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the outlives computation runs for now...

#![feature(rustc_attrs)]

//todo add all the test cases
// https://github.com/rust-lang/rfcs/blob/master/text/2093-infer-outlives.md#example-1-a-reference

#[rustc_outlives]
struct Direct<'a, T> { //~ ERROR 19:1: 21:2: [] [E0640]
    field: &'a T
}

fn main() { }
