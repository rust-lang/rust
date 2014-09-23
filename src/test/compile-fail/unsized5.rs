// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(struct_variant)]

// Test `Sized?` types not allowed in fields (except the last one).

struct S1<Sized? X> {
    f1: X, //~ ERROR `core::kinds::Sized` is not implemented
    f2: int,
}
struct S2<Sized? X> {
    f: int,
    g: X, //~ ERROR `core::kinds::Sized` is not implemented
    h: int,
}
struct S3 {
    f: str, //~ ERROR `core::kinds::Sized` is not implemented
    g: [uint]
}
struct S4 {
    f: str, //~ ERROR `core::kinds::Sized` is not implemented
    g: uint
}
enum E<Sized? X> {
    V1(X, int), //~ERROR `core::kinds::Sized` is not implemented
    V2{f1: X, f: int}, //~ERROR `core::kinds::Sized` is not implemented
}

pub fn main() {
}
