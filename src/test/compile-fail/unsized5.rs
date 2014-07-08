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

// Test `Sized?` types not allowed in fields.

struct S1<Sized? X> {
    f1: X, //~ ERROR type `f1` is dynamically sized. dynamically sized types may only appear as the
    f2: int,
}
struct S2<Sized? X> {
    f: int,
    g: X, //~ ERROR type `g` is dynamically sized. dynamically sized types may only appear as the ty
    h: int,
}

enum E<Sized? X> {
    V1(X, int), //~ERROR type `X` is dynamically sized. dynamically sized types may only appear as t
    V2{f1: X, f: int}, //~ERROR type `f1` is dynamically sized. dynamically sized types may only app
}

pub fn main() {
}
