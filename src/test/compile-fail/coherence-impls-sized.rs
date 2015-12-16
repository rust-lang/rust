// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(optin_builtin_traits)]

use std::marker::Copy;

enum TestE {
  A
}

struct MyType;

struct NotSync;
impl !Sync for NotSync {}

impl Sized for TestE {} //~ ERROR E0322

impl Sized for MyType {} //~ ERROR E0322

impl Sized for (MyType, MyType) {} //~ ERROR E0117

impl Sized for &'static NotSync {} //~ ERROR E0322

impl Sized for [MyType] {} //~ ERROR E0117

impl Sized for &'static [NotSync] {} //~ ERROR E0117

fn main() {
}
