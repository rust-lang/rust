// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(unboxed_closures, fn_traits)]

use std::ops::FnMut;

struct S {
    x: i32,
    y: i32,
}

impl FnMut<()> for S {
    extern "rust-call" fn call_mut(&mut self, (): ()) -> i32 {
        self.x * self.y
    }
}

impl FnOnce<()> for S {
    type Output = i32;
    extern "rust-call" fn call_once(mut self, args: ()) -> i32 { self.call_mut(args) }
}

fn main() {
    let mut s = S {
        x: 3,
        y: 3,
    };
    let ans = s();
    assert_eq!(ans, 9);
}
