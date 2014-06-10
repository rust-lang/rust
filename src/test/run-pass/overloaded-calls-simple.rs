// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(overloaded_calls)]

use std::ops::{Fn, FnMut, FnOnce};

struct S1 {
    x: int,
    y: int,
}

impl FnMut<(int,),int> for S1 {
    fn call_mut(&mut self, (z,): (int,)) -> int {
        self.x * self.y * z
    }
}

struct S2 {
    x: int,
    y: int,
}

impl Fn<(int,),int> for S2 {
    fn call(&self, (z,): (int,)) -> int {
        self.x * self.y * z
    }
}

struct S3 {
    x: int,
    y: int,
}

impl FnOnce<(int,int),int> for S3 {
    fn call_once(self, (z,zz): (int,int)) -> int {
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
    let ans = s(3);
    assert_eq!(ans, 27);

    let s = S3 {
        x: 3,
        y: 3,
    };
    let ans = s(3, 1);
    assert_eq!(ans, 27);
}

