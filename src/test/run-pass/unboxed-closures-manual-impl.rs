
// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closure_sugar)]

use std::ops::FnMut;

struct S;

impl FnMut<(int,),int> for S {
    extern "rust-call" fn call_mut(&mut self, (x,): (int,)) -> int {
        x * x
    }
}

fn call_it<F:FnMut(int)->int>(mut f: F, x: int) -> int {
    f.call_mut((x,)) + 3
}

fn call_box(f: &mut |&mut: int|->int, x: int) -> int {
    f.call_mut((x,)) + 3
}

fn main() {
    let x = call_it(S, 1);
    let y = call_box(&mut S, 1);
    assert!(x == 4);
    assert!(y == 4);
}

