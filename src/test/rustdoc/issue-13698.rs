// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-13698.rs

extern crate issue_13698;

pub struct Foo;
// @!has issue_13698/struct.Foo.html '//*[@id="method.foo"]' 'fn foo'
impl issue_13698::Foo for Foo {}

pub trait Bar {
    #[doc(hidden)]
    fn bar(&self) {}
}

// @!has issue_13698/struct.Foo.html '//*[@id="method.foo"]' 'fn bar'
impl Bar for Foo {}
