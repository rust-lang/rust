// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test equality constraints on associated types in a where clause.

pub trait ToInt {
    fn to_int(&self) -> int;
}

impl ToInt for int {
    fn to_int(&self) -> int { *self }
}

impl ToInt for uint {
    fn to_int(&self) -> int { *self as int }
}

pub trait GetToInt
{
    type R : ToInt;

    fn get(&self) -> <Self as GetToInt>::R;
}

impl GetToInt for int {
    type R = int;
    fn get(&self) -> int { *self }
}

impl GetToInt for uint {
    type R = uint;
    fn get(&self) -> uint { *self }
}

fn foo<G>(g: G) -> int
    where G : GetToInt
{
    ToInt::to_int(&g.get())
}

pub fn main() {
    assert_eq!(foo(22i), 22i);
    assert_eq!(foo(22u), 22i);
}
