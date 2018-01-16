// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that NLL analysis propagates lifetimes correctly through
// field accesses, Box accesses, etc.

#![feature(nll)]

fn foo(s: &mut (i32,)) -> i32 {
    let t = &mut *s; // this borrow should last for the entire function
    let x = &t.0;
    *s = (2,); //~ ERROR cannot assign to `*s`
    *x
}

fn bar(s: &Box<(i32,)>) -> &'static i32 {
    // FIXME(#46983): error message should be better
    &s.0 //~ ERROR explicit lifetime required in the type of `s` [E0621]
}

fn main() {
    foo(&mut (0,));
    bar(&Box::new((1,)));
}
