// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::Show;
use std::default::Default;

trait MyTrait {
    fn get(&self) -> Self;
}

impl<T> MyTrait for T
    where T : Default
{
    fn get(&self) -> T {
        Default::default()
    }
}

#[deriving(Clone,Show,PartialEq)]
struct MyType {
    dummy: uint
}

impl MyTrait for MyType {
    fn get(&self) -> MyType { (*self).clone() }
}

fn test_eq<M>(m: M, n: M)
where M : MyTrait + Show + PartialEq
{
    assert_eq!(m.get(), n);
}

pub fn main() {
    test_eq(0u, 0u);

    let value = MyType { dummy: 256 + 22 };
    test_eq(value, value);
}
