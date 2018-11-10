// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(trait_alias)]

trait Foo = PartialEq<i32> + Send;
trait Bar = Foo + Sync;

trait I32Iterator = Iterator<Item = i32>;

pub fn main() {
    let a: &dyn Bar = &123;
    assert!(*a == 123);
    let b = Box::new(456) as Box<dyn Foo>;
    assert!(*b == 456);

    let c: &mut dyn I32Iterator = &mut vec![123].into_iter();
    assert_eq!(c.next(), Some(123));
}
