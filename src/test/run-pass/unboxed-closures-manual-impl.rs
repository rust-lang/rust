// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closures, fn_traits)]

struct S;

impl FnMut<(i32,)> for S {
    extern "rust-call" fn call_mut(&mut self, (x,): (i32,)) -> i32 {
        x * x
    }
}

impl FnOnce<(i32,)> for S {
    type Output = i32;

    extern "rust-call" fn call_once(mut self, args: (i32,)) -> i32 { self.call_mut(args) }
}

fn call_it<F:FnMut(i32)->i32>(mut f: F, x: i32) -> i32 {
    f(x) + 3
}

fn call_box(f: &mut FnMut(i32) -> i32, x: i32) -> i32 {
    f(x) + 3
}

fn main() {
    let x = call_it(S, 1);
    let y = call_box(&mut S, 1);
    assert_eq!(x, 4);
    assert_eq!(y, 4);
}
