// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

// @has foo/struct.SomeStruct.html '//*[@class="sidebar-links"]/a[@href="#method.some_fn-1"]' \
//          "some_fn"
pub struct SomeStruct<T> { _inner: T }

impl SomeStruct<()> {
    pub fn some_fn(&self) {}
}

impl SomeStruct<usize> {
    pub fn some_fn(&self) {}
}
