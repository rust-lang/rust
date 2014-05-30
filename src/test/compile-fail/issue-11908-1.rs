// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-11908-1.rs
// ignore-android this test is incompatible with the android test runner
// error-pattern: multiple dylib candidates for `url` found

// This test ensures that if you have the same rlib or dylib at two locations
// in the same path that you don't hit an assertion in the compiler.
//
// Note that this relies on `liburl` to be in the path somewhere else,
// and then our aux-built libraries will collide with liburl (they have
// the same version listed)

extern crate url;

fn main() {}
