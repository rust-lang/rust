// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-19190-3.rs
// ignore-cross-compile

extern crate issue_19190_3;

use std::ops::Deref;
use issue_19190_3::Baz;

// @has issue_19190_3/struct.Foo.html
// @has - '//*[@id="method.as_str"]' 'fn as_str(&self) -> &str'
// @!has - '//*[@id="method.new"]' 'fn new() -> String'
pub use issue_19190_3::Foo;

// @has issue_19190_3/struct.Bar.html
// @has - '//*[@id="method.baz"]' 'fn baz(&self)'
// @!has - '//*[@id="method.static_baz"]' 'fn static_baz()'
pub use issue_19190_3::Bar;

// @has issue_19190_3/struct.MyBar.html
// @has - '//*[@id="method.baz"]' 'fn baz(&self)'
// @!has - '//*[@id="method.static_baz"]' 'fn static_baz()'
pub struct MyBar;

impl Deref for MyBar {
    type Target = Baz;
    fn deref(&self) -> &Baz { loop {} }
}

