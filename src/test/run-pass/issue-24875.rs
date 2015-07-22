// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(slice_patterns)]

fn main() {
    let v = vec![1, 2];
    assert_eq!(101, test(&v));
}

fn test(a: &[u64]) -> u64 {
    match a {
        [a, _b ..]  if a == 3  => 100,
        [a, _b]     if a == 1  => 101,
        [_a, _b ..]             => 102,
        _                     => 103
    }
}
