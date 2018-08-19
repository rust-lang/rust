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
// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

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
            drop(ra);
        }
        {
            let ra = &u.a;
            let a = u.a; // OK
            drop(ra);
        }
        {
            let ra = &u.a;
            let rma = &mut u.a; //[ast]~ ERROR cannot borrow `u.a` as mutable because it is also borrowed as immutable
                                //[mir]~^ ERROR cannot borrow `u.a` as mutable because it is also borrowed as immutable
            drop(ra);
        }
        {
            let ra = &u.a;
            u.a = 1; //[ast]~ ERROR cannot assign to `u.a` because it is borrowed
                     //[mir]~^ ERROR cannot assign to `u.a` because it is borrowed
            drop(ra);
        }
        // Imm borrow, other field
        {
            let ra = &u.a;
            let rb = &u.b; // OK
            drop(ra);
        }
        {
            let ra = &u.a;
            let b = u.b; // OK
            drop(ra);
        }
        {
            let ra = &u.a;
            let rmb = &mut u.b; //[ast]~ ERROR cannot borrow `u` (via `u.b`) as mutable because `u` is also borrowed as immutable (via `u.a`)
                                //[mir]~^ ERROR cannot borrow `u.b` as mutable because it is also borrowed as immutable
            drop(ra);
        }
        {
            let ra = &u.a;
            u.b = 1; //[ast]~ ERROR cannot assign to `u.b` because it is borrowed
                     //[mir]~^ ERROR cannot assign to `u.b` because it is borrowed
            drop(ra);
        }
        // Mut borrow, same field
        {
            let rma = &mut u.a;
            let ra = &u.a; //[ast]~ ERROR cannot borrow `u.a` as immutable because it is also borrowed as mutable
                         //[mir]~^ ERROR cannot borrow `u.a` as immutable because it is also borrowed as mutable
            drop(rma);
        }
        {
            let ra = &mut u.a;
            let a = u.a; //[ast]~ ERROR cannot use `u.a` because it was mutably borrowed
                         //[mir]~^ ERROR cannot use `u.a` because it was mutably borrowed
            drop(ra);
        }
        {
            let rma = &mut u.a;
            let rma2 = &mut u.a; //[ast]~ ERROR cannot borrow `u.a` as mutable more than once at a time
                                 //[mir]~^ ERROR cannot borrow `u.a` as mutable more than once at a time
            drop(rma);
        }
        {
            let rma = &mut u.a;
            u.a = 1; //[ast]~ ERROR cannot assign to `u.a` because it is borrowed
                     //[mir]~^ ERROR cannot assign to `u.a` because it is borrowed
            drop(rma);
        }
        // Mut borrow, other field
        {
            let rma = &mut u.a;
            let rb = &u.b; //[ast]~ ERROR cannot borrow `u` (via `u.b`) as immutable because `u` is also borrowed as mutable (via `u.a`)
                           //[mir]~^ ERROR cannot borrow `u.b` as immutable because it is also borrowed as mutable
            drop(rma);
        }
        {
            let ra = &mut u.a;
            let b = u.b; //[ast]~ ERROR cannot use `u.b` because it was mutably borrowed
                         //[mir]~^ ERROR cannot use `u.b` because it was mutably borrowed

            drop(ra);
        }
        {
            let rma = &mut u.a;
            let rmb2 = &mut u.b; //[ast]~ ERROR cannot borrow `u` (via `u.b`) as mutable more than once at a time
                                 //[mir]~^ ERROR cannot borrow `u.b` as mutable more than once at a time
            drop(rma);
        }
        {
            let rma = &mut u.a;
            u.b = 1; //[ast]~ ERROR cannot assign to `u.b` because it is borrowed
                     //[mir]~^ ERROR cannot assign to `u.b` because it is borrowed
            drop(rma);
        }
    }
}
