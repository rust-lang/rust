// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-11908-2.rs
// no-prefer-dynamic
// ignore-android this test is incompatible with the android test runner
// error-pattern: multiple rlib candidates for `url` found

// see comments in issue-11908-1 for what's going on here

extern crate url;

fn main() {}

