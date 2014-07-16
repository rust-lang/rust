// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:rlib_crate_test.rs
// ignore-stage1
// ignore-tidy-linelength
// ignore-android
// ignore-cross-compile gives a different error message

#![feature(phase)]
#[phase(plugin)] extern crate rlib_crate_test;
//~^ ERROR: plugin crate `rlib_crate_test` only found in rlib format, but must be available in dylib format

fn main() {}
