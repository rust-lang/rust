// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:derive-b.rs
// ignore-stage1

#[macro_use]
extern crate derive_b;

#[derive(Debug, PartialEq, B, Eq, Copy, Clone)]
#[B]
struct B {
    #[C]
    a: u64
}

fn main() {
    B { a: 3 };
    assert_eq!(B { a: 3 }, B { a: 3 });
    let b = B { a: 3 };
    let _d = b;
    let _e = b;
}
