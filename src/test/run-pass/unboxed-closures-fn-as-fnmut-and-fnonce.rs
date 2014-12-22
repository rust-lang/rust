// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks that the Fn trait hierarchy rules permit
// any Fn trait to be used where Fn is implemented.

#![feature(unboxed_closures)]
#![feature(unboxed_closures)]

use std::ops::{Fn,FnMut,FnOnce};

struct S;

impl Fn<(int,),int> for S {
    extern "rust-call" fn call(&self, (x,): (int,)) -> int {
        x * x
    }
}

fn call_it<F:Fn(int)->int>(f: &F, x: int) -> int {
    f.call((x,))
}

fn call_it_mut<F:FnMut(int)->int>(f: &mut F, x: int) -> int {
    f.call_mut((x,))
}

fn call_it_once<F:FnOnce(int)->int>(f: F, x: int) -> int {
    f.call_once((x,))
}

fn main() {
    let x = call_it(&S, 22);
    let y = call_it_mut(&mut S, 22);
    let z = call_it_once(S, 22);
    assert_eq!(x, y);
    assert_eq!(y, z);
}

