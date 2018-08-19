// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

use std::fmt;

// @!has foo/struct.Bar.html '//h3[@id="impl-ToString"]//code' 'impl<T> ToString for T'
pub struct Bar;

// @has foo/struct.Foo.html '//h3[@id="impl-ToString"]//code' 'impl<T> ToString for T'
pub struct Foo;
// @has foo/struct.Foo.html '//div[@class="sidebar-links"]/a[@href="#impl-ToString"]' 'ToString'

impl fmt::Display for Foo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Foo")
    }
}
