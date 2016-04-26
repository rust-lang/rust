// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-33113.rs
// build-aux-docs
// ignore-cross-compile

extern crate bar;

// @has issue_33113/trait.Bar.html
// @has - '//code' "for &'a char"
// @has - '//code' "for Foo"
pub use bar::Bar;
