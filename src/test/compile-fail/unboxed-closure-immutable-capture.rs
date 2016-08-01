// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that even unboxed closures that are capable of mutating their
// environment cannot mutate captured variables that have not been
// declared mutable (#18335)

fn set(x: &mut usize) { *x = 0; }

fn main() {
    let x = 0;
    move || x = 1; //~ ERROR cannot assign
    move || set(&mut x); //~ ERROR cannot borrow
    move || x = 1; //~ ERROR cannot assign
    move || set(&mut x); //~ ERROR cannot borrow
    || x = 1; //~ ERROR cannot assign
    // FIXME: this should be `cannot borrow` (issue #18330)
    || set(&mut x); //~ ERROR cannot assign
    || x = 1; //~ ERROR cannot assign
    // FIXME: this should be `cannot borrow` (issue #18330)
    || set(&mut x); //~ ERROR cannot assign
}
