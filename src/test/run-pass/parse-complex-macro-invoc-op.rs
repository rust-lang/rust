// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test parsing binary operators after macro invocations.

#![feature(macro_rules)]

macro_rules! id {
    ($e: expr) => { $e }
}

fn foo() {
    id!(1i) + 1;
    id![1i] - 1;
    id!(1i) * 1;
    id![1i] / 1;
    id!(1i) % 1;

    id!(1i) & 1;
    id![1i] | 1;
    id!(1i) ^ 1;

    let mut x = 1i;
    id![x] = 2;
    id!(x) += 1;

    id!(1f64).clone();

    id!([1i, 2, 3])[1];
    id![drop](1i);

    id!(true) && true;
    id![true] || true;
}

fn main() {}
