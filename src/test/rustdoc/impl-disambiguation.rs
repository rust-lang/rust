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

pub trait Foo {}

pub struct Bar<T> { field: T }

// @has foo/trait.Foo.html '//*[@class="item-list"]//code' \
//     "impl Foo for Bar<u8>"
impl Foo for Bar<u8> {}
// @has foo/trait.Foo.html '//*[@class="item-list"]//code' \
//     "impl Foo for Bar<u16>"
impl Foo for Bar<u16> {}
// @has foo/trait.Foo.html '//*[@class="item-list"]//code' \
//     "impl<'a> Foo for &'a Bar<u8>"
impl<'a> Foo for &'a Bar<u8> {}

pub mod mod1 {
    pub struct Baz {}
}

pub mod mod2 {
    pub enum Baz {}
}

// @has foo/trait.Foo.html '//*[@class="item-list"]//code' \
//     "impl Foo for foo::mod1::Baz"
impl Foo for mod1::Baz {}
// @has foo/trait.Foo.html '//*[@class="item-list"]//code' \
//     "impl<'a> Foo for &'a foo::mod2::Baz"
impl<'a> Foo for &'a mod2::Baz {}
