// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks that the Fn trait hierarchy rules do not permit
// Fn to be used where FnMut is implemented.

#![feature(fn_traits, unboxed_closures)]
#![feature(overloaded_calls)]

use std::ops::{Fn,FnMut,FnOnce};

struct S;

impl FnMut<(isize,)> for S {
    extern "rust-call" fn call_mut(&mut self, (x,): (isize,)) -> isize {
        x * x
    }
}

impl FnOnce<(isize,)> for S {
    type Output = isize;

    extern "rust-call" fn call_once(mut self, args: (isize,)) -> isize { self.call_mut(args) }
}

fn call_it<F:Fn(isize)->isize>(f: &F, x: isize) -> isize {
    f.call((x,))
}

fn main() {
    let x = call_it(&S, 22);
    //~^ ERROR E0277
}
