// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused)]
#![feature(nll)]

#[derive(Clone, Copy, Default)]
struct S {
    a: u8,
    b: u8,
}
#[derive(Clone, Copy, Default)]
struct Z {
    c: u8,
    d: u8,
}

union U {
    s: S,
    z: Z,
}

fn main() {
    unsafe {
        let mut u = U { s: Default::default() };

        let mref = &mut u.s.a;
        *mref = 22;

        let nref = &u.z.c;
        //~^ ERROR cannot borrow `u.z.c` as immutable because it is also borrowed as mutable [E0502]
        println!("{} {}", mref, nref)
    }
}

