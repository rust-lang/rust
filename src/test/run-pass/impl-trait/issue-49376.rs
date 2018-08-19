// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests for nested self-reference which caused a stack overflow.

use std::fmt::Debug;
use std::ops::*;

fn gen() -> impl PartialOrd + PartialEq + Debug { }

struct Bar {}
trait Foo<T = Self> {}
impl Foo for Bar {}

fn foo() -> impl Foo {
    Bar {}
}

fn test_impl_ops() -> impl Add + Sub + Mul + Div { 1 }
fn test_impl_assign_ops() -> impl AddAssign + SubAssign + MulAssign + DivAssign { 1 }

fn main() {}
