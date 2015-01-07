// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test slicing sugar.

#![feature(associated_types)]

extern crate core;
use core::ops::{Index, IndexMut, Range, RangeTo, RangeFrom, FullRange};

static mut COUNT: uint = 0;

struct Foo;

impl Index<Range<Foo>> for Foo {
    type Output = Foo;
    fn index(&self, index: &Range<Foo>) -> &Foo {
        unsafe { COUNT += 1; }
        self
    }
}
impl Index<RangeTo<Foo>> for Foo {
    type Output = Foo;
    fn index(&self, index: &RangeTo<Foo>) -> &Foo {
        unsafe { COUNT += 1; }
        self
    }
}
impl Index<RangeFrom<Foo>> for Foo {
    type Output = Foo;
    fn index(&self, index: &RangeFrom<Foo>) -> &Foo {
        unsafe { COUNT += 1; }
        self
    }
}
impl Index<FullRange> for Foo {
    type Output = Foo;
    fn index(&self, _index: &FullRange) -> &Foo {
        unsafe { COUNT += 1; }
        self
    }
}

impl IndexMut<Range<Foo>> for Foo {
    type Output = Foo;
    fn index_mut(&mut self, index: &Range<Foo>) -> &mut Foo {
        unsafe { COUNT += 1; }
        self
    }
}
impl IndexMut<RangeTo<Foo>> for Foo {
    type Output = Foo;
    fn index_mut(&mut self, index: &RangeTo<Foo>) -> &mut Foo {
        unsafe { COUNT += 1; }
        self
    }
}
impl IndexMut<RangeFrom<Foo>> for Foo {
    type Output = Foo;
    fn index_mut(&mut self, index: &RangeFrom<Foo>) -> &mut Foo {
        unsafe { COUNT += 1; }
        self
    }
}
impl IndexMut<FullRange> for Foo {
    type Output = Foo;
    fn index_mut(&mut self, _index: &FullRange) -> &mut Foo {
        unsafe { COUNT += 1; }
        self
    }
}


fn main() {
    let mut x = Foo;
    &x[];
    &x[Foo..];
    &x[..Foo];
    &x[Foo..Foo];
    &mut x[];
    &mut x[Foo..];
    &mut x[..Foo];
    &mut x[Foo..Foo];
    unsafe {
        assert!(COUNT == 8);
    }
}
