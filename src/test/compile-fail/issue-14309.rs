// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(ctypes)]
#![allow(dead_code)]

struct A { //~ NOTE consider adding `#[repr(C)]` to this type
    x: int
}

#[repr(C, packed)]
struct B {
    x: int,
    y: A
}

#[repr(C)]
struct C {
    x: int
}

type A2 = A;
type B2 = B;
type C2 = C;

#[repr(C)]
struct D {
    x: C,
    y: A
}

extern "C" {
    fn foo(x: A); //~ ERROR found struct without FFI-safe representation used in FFI
    fn bar(x: B); //~ ERROR FFI-safe
    fn baz(x: C);
    fn qux(x: A2); //~ ERROR FFI-safe
    fn quux(x: B2); //~ ERROR FFI-safe
    fn corge(x: C2);
    fn fred(x: D); //~ ERROR FFI-safe
}

fn main() { }
