// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{
    ops::Deref,
};

struct Ptr<T: ?Sized>(Box<T>);

impl<T: ?Sized> Deref for Ptr<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &*self.0
    }
}

trait Foo {
    fn foo(self: Ptr<Self>); //~ ERROR `Ptr<Self>` cannot be used as the type of `self` without
}

struct Bar;

impl Foo for Bar {
    fn foo(self: Ptr<Self>) {} //~ ERROR `Ptr<Bar>` cannot be used as the type of `self` without
}

impl Bar {
    fn bar(self: Box<Ptr<Self>>) {} //~ ERROR `std::boxed::Box<Ptr<Bar>>` cannot be used as the
}

fn main() {}
