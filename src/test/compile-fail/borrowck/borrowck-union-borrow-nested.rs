// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

#![feature(untagged_unions)]

#[derive(Clone, Copy)]
struct S {
    a: u8,
    b: u16,
}

#[derive(Clone, Copy)]
union U {
    s: S,
    c: u32,
}

fn main() {
    unsafe {
        {
            let mut u = U { s: S { a: 0, b: 1 } };
            let ra = &mut u.s.a;
            let b = u.s.b; // OK
        }
        {
            let mut u = U { s: S { a: 0, b: 1 } };
            let ra = &mut u.s.a;
            let b = u.c; //~ ERROR cannot use `u.c` because it was mutably borrowed
        }
    }
}
