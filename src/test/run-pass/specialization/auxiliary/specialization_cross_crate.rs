// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(specialization)]

pub trait Foo {
    fn foo(&self) -> &'static str;
}

impl<T> Foo for T {
    default fn foo(&self) -> &'static str {
        "generic"
    }
}

impl<T: Clone> Foo for T {
    default fn foo(&self) -> &'static str {
        "generic Clone"
    }
}

impl<T, U> Foo for (T, U) where T: Clone, U: Clone {
    default fn foo(&self) -> &'static str {
        "generic pair"
    }
}

impl<T: Clone> Foo for (T, T) {
    default fn foo(&self) -> &'static str {
        "generic uniform pair"
    }
}

impl Foo for (u8, u32) {
    default fn foo(&self) -> &'static str {
        "(u8, u32)"
    }
}

impl Foo for (u8, u8) {
    default fn foo(&self) -> &'static str {
        "(u8, u8)"
    }
}

impl<T: Clone> Foo for Vec<T> {
    default fn foo(&self) -> &'static str {
        "generic Vec"
    }
}

impl Foo for Vec<i32> {
    fn foo(&self) -> &'static str {
        "Vec<i32>"
    }
}

impl Foo for String {
    fn foo(&self) -> &'static str {
        "String"
    }
}

impl Foo for i32 {
    fn foo(&self) -> &'static str {
        "i32"
    }
}

pub trait MyMarker {}
impl<T: Clone + MyMarker> Foo for T {
    default fn foo(&self) -> &'static str {
        "generic Clone + MyMarker"
    }
}
