// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(lang_items, unboxed_closures, fn_traits)]

use std::ops::{Fn, FnMut, FnOnce};

struct S1 {
    x: i32,
    y: i32,
}

impl FnMut<(i32,)> for S1 {
    extern "rust-call" fn call_mut(&mut self, (z,): (i32,)) -> i32 {
        self.x * self.y * z
    }
}

impl FnOnce<(i32,)> for S1 {
    type Output = i32;
    extern "rust-call" fn call_once(mut self, args: (i32,)) -> i32 {
        self.call_mut(args)
    }
}

struct S2 {
    x: i32,
    y: i32,
}

impl Fn<(i32,)> for S2 {
    extern "rust-call" fn call(&self, (z,): (i32,)) -> i32 {
        self.x * self.y * z
    }
}

impl FnMut<(i32,)> for S2 {
    extern "rust-call" fn call_mut(&mut self, args: (i32,)) -> i32 { self.call(args) }
}

impl FnOnce<(i32,)> for S2 {
    type Output = i32;
    extern "rust-call" fn call_once(self, args: (i32,)) -> i32 { self.call(args) }
}

struct S3 {
    x: i32,
    y: i32,
}

impl FnOnce<(i32,i32)> for S3 {
    type Output = i32;
    extern "rust-call" fn call_once(self, (z,zz): (i32,i32)) -> i32 {
        self.x * self.y * z * zz
    }
}

fn main() {
    let mut s = S1 {
        x: 3,
        y: 3,
    };
    let ans = s(3);

    assert_eq!(ans, 27);
    let s = S2 {
        x: 3,
        y: 3,
    };
    let ans = s.call((3,));
    assert_eq!(ans, 27);

    let s = S3 {
        x: 3,
        y: 3,
    };
    let ans = s(3, 1);
    assert_eq!(ans, 27);
}
