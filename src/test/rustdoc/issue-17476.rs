// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-17476.rs
// ignore-android

extern crate issue_17476;

pub struct Foo;

// @has issue_17476/struct.Foo.html \
//      '//*[@href="http://example.com/issue_17476/trait.Foo.html#method.foo"]' \
//      'foo'
impl issue_17476::Foo for Foo {}
