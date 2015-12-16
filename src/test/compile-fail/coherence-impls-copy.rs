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

impl Copy for TestE {}
impl Clone for TestE { fn clone(&self) -> Self { *self } }

impl Copy for MyType {}

impl Copy for &'static mut MyType {}
//~^ ERROR E0206
impl Clone for MyType { fn clone(&self) -> Self { *self } }

impl Copy for (MyType, MyType) {}
//~^ ERROR E0206
//~| ERROR E0117

impl Copy for &'static NotSync {}
//~^ ERROR E0206

impl Copy for [MyType] {}
//~^ ERROR E0206
//~| ERROR E0117

impl Copy for &'static [NotSync] {}
//~^ ERROR E0206
//~| ERROR E0117

fn main() {
}
