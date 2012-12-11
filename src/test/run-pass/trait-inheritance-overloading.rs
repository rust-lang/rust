// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cmp::Eq;

trait MyNum : Add<self,self> Sub<self,self> Mul<self,self> Eq { }

struct MyInt { val: int }

impl MyInt : Add<MyInt, MyInt> {
    pure fn add(&self, other: &MyInt) -> MyInt { mi(self.val + other.val) }
}

impl MyInt : Sub<MyInt, MyInt> {
    pure fn sub(&self, other: &MyInt) -> MyInt { mi(self.val - other.val) }
}

impl MyInt : Mul<MyInt, MyInt> {
    pure fn mul(&self, other: &MyInt) -> MyInt { mi(self.val * other.val) }
}

impl MyInt : Eq {
    pure fn eq(&self, other: &MyInt) -> bool { self.val == other.val }
    pure fn ne(&self, other: &MyInt) -> bool { !self.eq(other) }
}

impl MyInt : MyNum;

fn f<T:Copy MyNum>(x: T, y: T) -> (T, T, T) {
    return (x + y, x - y, x * y);
}

pure fn mi(v: int) -> MyInt { MyInt { val: v } }

fn main() {
    let (x, y) = (mi(3), mi(5));
    let (a, b, c) = f(x, y);
    assert a == mi(8);
    assert b == mi(-2);
    assert c == mi(15);
}

