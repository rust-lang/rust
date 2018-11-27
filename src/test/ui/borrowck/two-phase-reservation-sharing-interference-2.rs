// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a reservation conflicts with a shared borrow.
//
// We conservatively treat reservation as creating a mutable reference, which
// should be unique, and then allowing sharing up to the activation point.

#![feature(rustc_attrs)]

fn main() {
    let mut v = vec![0, 1, 2];
    let shared = &v;

    v.push(shared.len()); //~ ERROR cannot borrow `v` as mutable

    assert_eq!(v, [0, 1, 2, 3]);
}
