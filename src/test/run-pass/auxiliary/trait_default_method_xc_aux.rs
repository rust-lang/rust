// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct Something { pub x: isize }

pub trait A {
    fn f(&self) -> isize;
    fn g(&self) -> isize { 10 }
    fn h(&self) -> isize { 11 }
    fn lurr(x: &Self, y: &Self) -> isize { x.g() + y.h() }
}


impl A for isize {
    fn f(&self) -> isize { 10 }
}

impl A for Something {
    fn f(&self) -> isize { 10 }
}

pub trait B<T> {
    fn thing<U>(&self, x: T, y: U) -> (T, U) { (x, y) }
    fn staticthing<U>(_z: &Self, x: T, y: U) -> (T, U) { (x, y) }
}

impl<T> B<T> for isize { }
impl B<f64> for bool { }



pub trait TestEquality {
    fn test_eq(&self, rhs: &Self) -> bool;
    fn test_neq(&self, rhs: &Self) -> bool {
        !self.test_eq(rhs)
    }
}

impl TestEquality for isize {
    fn test_eq(&self, rhs: &isize) -> bool {
        *self == *rhs
    }
}
