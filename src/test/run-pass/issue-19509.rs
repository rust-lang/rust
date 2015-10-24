// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Deref;

struct Foo {
    inner: Bar,
}

struct Bar;

impl Foo {
    pub fn foo_method(&self) {
    }
}

impl Bar {
    pub fn bar_method(&self) {
    }
}

impl Deref for Foo {
    type Target = Bar;

    fn deref<'a>(&'a self) -> &'a Self::Target {
        &self.inner
    }
}

impl Deref for Bar {
    type Target = Foo;

    fn deref<'a>(&'a self) -> &'a Self::Target {
        panic!()
    }
}

fn main() {
    let foo = Foo { inner: Bar, };
    let bar = Bar;

    foo.bar_method();    // should compile and execute
}
