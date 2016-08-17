// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// @has issue_25001/struct.Foo.html
pub struct Foo<T>(T);

pub trait Bar {
    type Item;

    fn quux(self);
}

impl Foo<u8> {
    // @has - '//*[@id="method.pass"]//code' 'fn pass()'
    // @has - '//*[@id="pass.v"]//code' 'fn pass()'
    pub fn pass() {}
}
impl Foo<u16> {
    // @has - '//*[@id="method.pass-1"]//code' 'fn pass() -> usize'
    // @has - '//*[@id="pass.v-1"]//code' 'fn pass() -> usize'
    pub fn pass() -> usize { 42 }
}
impl Foo<u32> {
    // @has - '//*[@id="method.pass-2"]//code' 'fn pass() -> isize'
    // @has - '//*[@id="pass.v-2"]//code' 'fn pass() -> isize'
    pub fn pass() -> isize { 42 }
}

impl<T> Bar for Foo<T> {
    // @has - '//*[@id="associatedtype.Item"]//code' 'type Item = T'
    type Item=T;

    // @has - '//*[@id="method.quux"]//code' 'fn quux(self)'
    fn quux(self) {}
}
impl<'a, T> Bar for &'a Foo<T> {
    // @has - '//*[@id="associatedtype.Item-1"]//code' "type Item = &'a T"
    type Item=&'a T;

    // @has - '//*[@id="method.quux-1"]//code' 'fn quux(self)'
    fn quux(self) {}
}
impl<'a, T> Bar for &'a mut Foo<T> {
    // @has - '//*[@id="associatedtype.Item-2"]//code' "type Item = &'a mut T"
    type Item=&'a mut T;

    // @has - '//*[@id="method.quux-2"]//code' 'fn quux(self)'
    fn quux(self) {}
}
