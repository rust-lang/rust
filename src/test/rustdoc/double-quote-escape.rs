// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

// ignore-tidy-linelength

pub trait Foo<T> {
    fn foo() {}
}

pub struct Bar;

// @has foo/struct.Bar.html
// @has - '//*[@class="sidebar-links"]/a[@href="#impl-Foo%3Cunsafe%20extern%20%22C%22%20fn()%3E"]' 'Foo<unsafe extern "C" fn()>'
impl Foo<unsafe extern "C" fn()> for Bar {}
