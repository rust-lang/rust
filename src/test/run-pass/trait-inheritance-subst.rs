// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub trait Add<RHS,Result> {
    pure fn add(rhs: &RHS) -> Result;
}

trait MyNum : Add<Self,Self> { }

struct MyInt { val: int }

impl Add<MyInt, MyInt> for MyInt {
    pure fn add(other: &MyInt) -> MyInt { mi(self.val + other.val) }
}

impl MyNum for MyInt;

fn f<T:MyNum>(x: T, y: T) -> T {
    return x.add(&y);
}

pure fn mi(v: int) -> MyInt { MyInt { val: v } }

pub fn main() {
    let (x, y) = (mi(3), mi(5));
    let z = f(x, y);
    fail_unless!(z.val == 8)
}

