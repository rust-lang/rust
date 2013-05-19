// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::cmp::Eq;

trait MyNum : Eq { }

struct MyInt { val: int }

impl Eq for MyInt {
    fn eq(&self, other: &MyInt) -> bool { self.val == other.val }
    fn ne(&self, other: &MyInt) -> bool { !self.eq(other) }
}

impl MyNum for MyInt;

fn f<T:MyNum>(x: T, y: T) -> bool {
    return x == y;
}

fn mi(v: int) -> MyInt { MyInt { val: v } }

pub fn main() {
    let (x, y, z) = (mi(3), mi(5), mi(3));
    assert!(x != y);
    assert_eq!(x, z);
}
