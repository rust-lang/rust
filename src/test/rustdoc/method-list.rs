// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

#![crate_name = "foo"]

// @has foo/struct.Foo.html
// @has - '//*[@class="sidebar-links"]/a' 'super_long_name'
// @has - '//*[@class="sidebar-links"]/a' 'Disp'
pub struct Foo(usize);

impl Foo {
    pub fn super_long_name() {}
}

pub trait Disp {
    fn disp_trait_method();
}

impl Disp for Foo {
    fn disp_trait_method() {}
}
