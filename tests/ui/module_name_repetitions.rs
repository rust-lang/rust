// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::module_name_repetitions)]
#![allow(dead_code)]

mod foo {
    pub fn foo() {}
    pub fn foo_bar() {}
    pub fn bar_foo() {}
    pub struct FooCake {}
    pub enum CakeFoo {}
    pub struct Foo7Bar;

    // Should not warn
    pub struct Foobar;
}

fn main() {}
