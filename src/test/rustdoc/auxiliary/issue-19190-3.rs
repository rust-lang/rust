// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Cmetadata=aux

use std::ops::Deref;

pub struct Foo;

impl Deref for Foo {
    type Target = i32;
    fn deref(&self) -> &i32 { loop {} }
}

pub struct Bar;
pub struct Baz;

impl Baz {
    pub fn baz(&self) {}
    pub fn static_baz() {}
}

impl Deref for Bar {
    type Target = Baz;
    fn deref(&self) -> &Baz { loop {} }
}
