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

pub trait Bar<T, U> {}

// @has 'foo/struct.Foo1.html'
pub struct Foo1;
// @count - '//*[@class="impl"]' 1
// @has - '//*[@class="impl"]' "impl Bar<Foo1, &'static Foo1> for Foo1"
impl Bar<Foo1, &'static Foo1> for Foo1 {}

// @has 'foo/struct.Foo2.html'
pub struct Foo2;
// @count - '//*[@class="impl"]' 1
// @has - '//*[@class="impl"]' "impl Bar<&'static Foo2, Foo2> for u8"
impl Bar<&'static Foo2, Foo2> for u8 {}
