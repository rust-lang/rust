// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_type_defaults)]

trait Foo<T> {
    type Out = T;
    fn foo(&self) -> Self::Out;
}

impl Foo<u32> for () {
    fn foo(&self) -> u32 {
        4u32
    }
}

impl Foo<u64> for bool {
    type Out = ();
    fn foo(&self) {}
}

fn main() {
    assert_eq!(<() as Foo<u32>>::foo(&()), 4u32);
    assert_eq!(<bool as Foo<u64>>::foo(&true), ());
}
