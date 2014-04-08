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

// Test `type` types not allowed in fields or local variables.

/*trait T for type {}

fn f5<type X>(x: &X) {
    let _: X; // ERROR local variable with dynamically sized type X
    let _: (int, (X, int)); // ERROR local variable with dynamically sized type (int,(X,int))
}
fn f6<type X: T>(x: &X) {
    let _: X; // ERROR local variable with dynamically sized type X
    let _: (int, (X, int)); // ERROR local variable with dynamically sized type (int,(X,int))
}*/

struct S1<type X> {
    f1: X, //~ ERROR type of field f1 is dynamically sized
    f2: int,
}
struct S2<type X> {
    f: int,
    g: X, //~ ERROR type of field g is dynamically sized
    h: int,
}

enum E<type X> {
    V1(X, int), //~ERROR type X is dynamically sized
    V2{f1: X, f: int}, //~ERROR type of field f1 is dynamically sized
}

pub fn main() {
}
