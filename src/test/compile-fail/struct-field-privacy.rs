// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:struct-field-privacy.rs

extern crate xc = "struct-field-privacy";

struct A {
    a: int,
}

mod inner {
    struct A {
        a: int,
        pub b: int,
        priv c: int, //~ ERROR: unnecessary `priv` visibility
    }
    pub struct B {
        a: int,
        priv b: int,
        pub c: int,
    }
}

fn test(a: A, b: inner::A, c: inner::B, d: xc::A, e: xc::B) {
    //~^ ERROR: type `A` is private
    //~^^ ERROR: struct `A` is private

    a.a;
    b.a; //~ ERROR: field `a` is private
    b.b;
    b.c; //~ ERROR: field `c` is private
    c.a;
    c.b; //~ ERROR: field `b` is private
    c.c;

    d.a; //~ ERROR: field `a` is private
    d.b;

    e.a;
    e.b; //~ ERROR: field `b` is private
}

fn main() {}
