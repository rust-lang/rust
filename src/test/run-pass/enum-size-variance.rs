// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-pretty

#![deny(enum_size_variance)]
#![allow(dead_code)]

enum Enum1 { }

enum Enum2 { A, B, C }

enum Enum3 { D(int), E, F }

enum Enum4 { H(int), I(int), J }

enum Enum5 { //~ ERROR three times larger
    L(int, int, int, int), //~ NOTE this variant is the largest
    M(int),
    N
}

enum Enum6<T, U> {
    O(T),
    P(U),
    Q(int)
}

#[allow(enum_size_variance)]
enum Enum7 {
    R(int, int, int, int),
    S(int),
    T
}
pub fn main() { }
