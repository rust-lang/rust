// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:derive-a.rs
// ignore-stage1

#[macro_use]
extern crate derive_a;

#[derive(Debug, PartialEq, A, Eq, Copy, Clone)]
struct A;

fn main() {
    A;
    assert_eq!(A, A);
    A.clone();
    let a = A;
    let _c = a;
    let _d = a;
}
