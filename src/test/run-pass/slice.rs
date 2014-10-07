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

#![feature(slicing_syntax)]

extern crate core;
use core::ops::{Slice,SliceMut};

static mut COUNT: uint = 0;

struct Foo;

impl Slice<Foo, Foo> for Foo {
    fn as_slice_<'a>(&'a self) -> &'a Foo {
        unsafe { COUNT += 1; }
        self
    }
    fn slice_from_or_fail<'a>(&'a self, _from: &Foo) -> &'a Foo {
        unsafe { COUNT += 1; }
        self
    }
    fn slice_to_or_fail<'a>(&'a self, _to: &Foo) -> &'a Foo {
        unsafe { COUNT += 1; }
        self
    }
    fn slice_or_fail<'a>(&'a self, _from: &Foo, _to: &Foo) -> &'a Foo {
        unsafe { COUNT += 1; }
        self
    }
}

impl SliceMut<Foo, Foo> for Foo {
    fn as_mut_slice_<'a>(&'a mut self) -> &'a mut Foo {
        unsafe { COUNT += 1; }
        self
    }
    fn slice_from_or_fail_mut<'a>(&'a mut self, _from: &Foo) -> &'a mut Foo {
        unsafe { COUNT += 1; }
        self
    }
    fn slice_to_or_fail_mut<'a>(&'a mut self, _to: &Foo) -> &'a mut Foo {
        unsafe { COUNT += 1; }
        self
    }
    fn slice_or_fail_mut<'a>(&'a mut self, _from: &Foo, _to: &Foo) -> &'a mut Foo {
        unsafe { COUNT += 1; }
        self
    }
}
fn main() {
    let mut x = Foo;
    x[];
    x[Foo..];
    x[..Foo];
    x[Foo..Foo];
    x[mut];
    x[mut Foo..];
    x[mut ..Foo];
    x[mut Foo..Foo];
    unsafe {
        assert!(COUNT == 8);
    }
}
