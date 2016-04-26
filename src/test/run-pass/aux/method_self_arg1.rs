// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "lib"]

#![allow(unknown_features)]
#![feature(box_syntax)]

static mut COUNT: u64 = 1;

pub fn get_count() -> u64 { unsafe { COUNT } }

#[derive(Copy, Clone)]
pub struct Foo;

impl Foo {
    pub fn foo(self, x: &Foo) {
        unsafe { COUNT *= 2; }
        // Test internal call.
        Foo::bar(&self);
        Foo::bar(x);

        Foo::baz(self);
        Foo::baz(*x);

        Foo::qux(box self);
        Foo::qux(box *x);
    }

    pub fn bar(&self) {
        unsafe { COUNT *= 3; }
    }

    pub fn baz(self) {
        unsafe { COUNT *= 5; }
    }

    pub fn qux(self: Box<Foo>) {
        unsafe { COUNT *= 7; }
    }
}
