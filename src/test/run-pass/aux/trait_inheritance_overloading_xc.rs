// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::PartialEq;
use std::ops::{Add, Sub, Mul};

pub trait MyNum : Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self> + PartialEq + Clone {
}

#[derive(Clone, Debug)]
pub struct MyInt {
    pub val: isize
}

impl Add for MyInt {
    type Output = MyInt;

    fn add(self, other: MyInt) -> MyInt { mi(self.val + other.val) }
}

impl Sub for MyInt {
    type Output = MyInt;

    fn sub(self, other: MyInt) -> MyInt { mi(self.val - other.val) }
}

impl Mul for MyInt {
    type Output = MyInt;

    fn mul(self, other: MyInt) -> MyInt { mi(self.val * other.val) }
}

impl PartialEq for MyInt {
    fn eq(&self, other: &MyInt) -> bool { self.val == other.val }

    fn ne(&self, other: &MyInt) -> bool { !self.eq(other) }
}

impl MyNum for MyInt {}

fn mi(v: isize) -> MyInt { MyInt { val: v } }
