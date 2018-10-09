// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

// This is a stub feature that doesn't control anything, so to make tidy happy,
// gate-test-test_2018_feature

#![feature(test_2018_feature)]
//~^ WARN the feature `test_2018_feature` is included in the Rust 2018 edition
#![feature(rust_2018_preview)]

fn main() {}
