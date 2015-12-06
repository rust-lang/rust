// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-15318.rs
// ignore-cross-compile

#![no_std]

extern crate issue_15318;

// @has issue_15318/fn.bar.html \
//      '//*[@href="http://example.com/issue_15318/primitive.pointer.html"]' \
//      '*mut T'
pub fn bar<T>(ptr: *mut T) {}
