// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-21801.rs

extern crate issue_21801;

// @has issue_21801/struct.Foo.html
// @has - '//*[@id="method.new"]' \
//        'fn new<F>(f: F) -> Foo where F: FnMut() -> i32'
pub use issue_21801::Foo;
