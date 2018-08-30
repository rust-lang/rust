// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::{Deref, DerefMut};

struct Foo;

impl Foo {
    fn foo_mut(&mut self) {}
}

struct Bar(Foo);

impl Deref for Bar {
    type Target = Foo;

    fn deref(&self) -> &Foo {
        &self.0
    }
}

impl DerefMut for Bar {
    fn deref_mut(&mut self) -> &mut Foo {
        &mut self.0
    }
}

fn test(mut bar: Box<Bar>) {
    bar.foo_mut();
}

fn main() {}
