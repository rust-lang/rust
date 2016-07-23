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

// @has issue_19190_3/Foo.t.html
// @has - '//*[@id="count_ones.v"]' 'fn count_ones(self) -> u32'
// @!has - '//*[@id="min_value.v"]' 'fn min_value() -> i32'
pub use issue_19190_3::Foo;

// @has issue_19190_3/Bar.t.html
// @has - '//*[@id="baz.v"]' 'fn baz(&self)'
// @!has - '//*[@id="static_baz.v"]' 'fn static_baz()'
pub use issue_19190_3::Bar;

// @has issue_19190_3/MyBar.t.html
// @has - '//*[@id="baz.v"]' 'fn baz(&self)'
// @!has - '//*[@id="static_baz.v"]' 'fn static_baz()'
pub struct MyBar;

impl Deref for MyBar {
    type Target = Baz;
    fn deref(&self) -> &Baz { loop {} }
}

