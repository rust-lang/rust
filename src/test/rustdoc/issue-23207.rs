// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-23207-1.rs
// aux-build:issue-23207-2.rs
// ignore-cross-compile

extern crate issue_23207_2;

// @has issue_23207/fmt/index.html
// @count - '//*[@class="struct"]' 1
pub use issue_23207_2::fmt;

