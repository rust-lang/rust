// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo<B> {
    fn bar(&self){}
}

impl Foo<u8> for i8 {}
impl Foo<u16> for i8 {}
impl Foo<u32> for i8 {}
impl Foo<u64> for i8 {}
impl Foo<bool> for i8 {}

impl Foo<u16> for u8 {}
impl Foo<u32> for u8 {}
impl Foo<u64> for u8 {}
impl Foo<bool> for u8 {}

impl Foo<u8> for bool {}
impl Foo<u16> for bool {}
impl Foo<u32> for bool {}
impl Foo<u64> for bool {}
impl Foo<bool> for bool {}
impl Foo<i8> for bool {}

fn main() {
    Foo::<i32>::bar(&1i8);
    Foo::<i32>::bar(&1u8);
    Foo::<i32>::bar(&true);
}
