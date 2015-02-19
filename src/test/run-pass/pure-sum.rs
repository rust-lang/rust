// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that functions can modify local state.

#![allow(unknown_features)]
#![feature(box_syntax)]

fn sums_to(v: Vec<int> , sum: int) -> bool {
    let mut i = 0_usize;
    let mut sum0 = 0;
    while i < v.len() {
        sum0 += v[i];
        i += 1_usize;
    }
    return sum0 == sum;
}

fn sums_to_using_uniq(v: Vec<int> , sum: int) -> bool {
    let mut i = 0_usize;
    let mut sum0 = box 0;
    while i < v.len() {
        *sum0 += v[i];
        i += 1_usize;
    }
    return *sum0 == sum;
}

fn sums_to_using_rec(v: Vec<int> , sum: int) -> bool {
    let mut i = 0_usize;
    let mut sum0 = F {f: 0};
    while i < v.len() {
        sum0.f += v[i];
        i += 1_usize;
    }
    return sum0.f == sum;
}

struct F<T> { f: T }

fn sums_to_using_uniq_rec(v: Vec<int> , sum: int) -> bool {
    let mut i = 0_usize;
    let mut sum0 = F {f: box 0};
    while i < v.len() {
        *sum0.f += v[i];
        i += 1_usize;
    }
    return *sum0.f == sum;
}

pub fn main() {
}
