// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that enabling an unstable feature disables warnings

// aux-build:stability_cfg2.rs

#![feature(unstable_test_feature)]
#![deny(non_snake_case)] // To trigger a hard error

// Shouldn't generate a warning about unstable features
#[allow(unused_extern_crates)]
extern crate stability_cfg2;

pub fn BOGUS() { } //~ ERROR

pub fn main() { }
