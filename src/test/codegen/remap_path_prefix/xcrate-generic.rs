// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-windows
// ignore-tidy-linelength
// compile-flags: -g -C metadata=foo -C no-prepopulate-passes
// aux-build:xcrate-generic.rs

#![crate_type = "lib"]

extern crate xcrate_generic;

pub fn foo() {
    xcrate_generic::foo::<u32>();
}

// Here we check that local debuginfo is mapped correctly.
// CHECK: !DIFile(filename: "/the/aux-src/xcrate-generic.rs", directory: "")
