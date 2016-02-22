// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:struct_field_privacy.rs

extern crate struct_field_privacy as xc;

struct A {
    a: isize,
}

mod inner {
    pub struct A {
        a: isize,
        pub b: isize,
    }
    pub struct B {
        pub a: isize,
        b: isize,
    }
}

fn test(a: A, b: inner::A, c: inner::B, d: xc::A, e: xc::B) {
    a.a;
    b.a; //~ ERROR: field `a` of struct `inner::A` is private
    b.b;
    c.a;
    c.b; //~ ERROR: field `b` of struct `inner::B` is private

    d.a; //~ ERROR: field `a` of struct `xc::A` is private
    d.b;

    e.a;
    e.b; //~ ERROR: field `b` of struct `xc::B` is private
}

fn main() {}
