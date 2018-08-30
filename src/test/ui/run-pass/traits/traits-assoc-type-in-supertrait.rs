// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test case where an associated type is referenced from within the
// supertrait definition. Issue #20220.


use std::vec::IntoIter;

pub trait Foo: Iterator<Item=<Self as Foo>::Key> {
    type Key;
}

impl Foo for IntoIter<i32> {
    type Key = i32;
}

fn sum_foo<F:Foo<Key=i32>>(f: F) -> i32 {
    f.fold(0, |a,b| a + b)
}

fn main() {
    let x = sum_foo(vec![11, 10, 1].into_iter());
    assert_eq!(x, 22);
}
