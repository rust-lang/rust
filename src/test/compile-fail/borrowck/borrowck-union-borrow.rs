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
union U {
    a: u8,
    b: u64,
}

fn main() {
    unsafe {
        let mut u = U { b: 0 };
        // Imm borrow, same field
        {
            let ra = &u.a;
            let ra2 = &u.a; // OK
        }
        {
            let ra = &u.a;
            let a = u.a; // OK
        }
        {
            let ra = &u.a;
            let rma = &mut u.a; //~ ERROR cannot borrow `u.a` as mutable because it is also borrowed as immutable
        }
        {
            let ra = &u.a;
            u.a = 1; //~ ERROR cannot assign to `u.a` because it is borrowed
        }
        // Imm borrow, other field
        {
            let ra = &u.a;
            let rb = &u.b; // OK
        }
        {
            let ra = &u.a;
            let b = u.b; // OK
        }
        {
            let ra = &u.a;
            let rmb = &mut u.b; //~ ERROR cannot borrow `u` (via `u.b`) as mutable because `u` is also borrowed as immutable (via `u.a`)
        }
        {
            let ra = &u.a;
            u.b = 1; //~ ERROR cannot assign to `u.b` because it is borrowed
        }
        // Mut borrow, same field
        {
            let rma = &mut u.a;
            let ra = &u.a; //~ ERROR cannot borrow `u.a` as immutable because it is also borrowed as mutable
        }
        {
            let ra = &mut u.a;
            let a = u.a; //~ ERROR cannot use `u.a` because it was mutably borrowed
        }
        {
            let rma = &mut u.a;
            let rma2 = &mut u.a; //~ ERROR cannot borrow `u.a` as mutable more than once at a time
        }
        {
            let rma = &mut u.a;
            u.a = 1; //~ ERROR cannot assign to `u.a` because it is borrowed
        }
        // Mut borrow, other field
        {
            let rma = &mut u.a;
            let rb = &u.b; //~ ERROR cannot borrow `u` (via `u.b`) as immutable because `u` is also borrowed as mutable (via `u.a`)
        }
        {
            let ra = &mut u.a;
            let b = u.b; //~ ERROR cannot use `u.b` because it was mutably borrowed
        }
        {
            let rma = &mut u.a;
            let rmb2 = &mut u.b; //~ ERROR cannot borrow `u` (via `u.b`) as mutable more than once at a time
        }
        {
            let rma = &mut u.a;
            u.b = 1; //~ ERROR cannot assign to `u.b` because it is borrowed
        }
    }
}
