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
//~^ ERROR the trait `Copy` may not be implemented for this type
//~| NOTE type is not a structure or enumeration
impl Clone for MyType { fn clone(&self) -> Self { *self } }

impl Copy for (MyType, MyType) {}
//~^ ERROR the trait `Copy` may not be implemented for this type
//~| NOTE type is not a structure or enumeration
//~| ERROR only traits defined in the current crate can be implemented for arbitrary types
//~| NOTE impl doesn't use types inside crate
//~| NOTE the impl does not reference any types defined in this crate

impl Copy for &'static NotSync {}
//~^ ERROR the trait `Copy` may not be implemented for this type
//~| NOTE type is not a structure or enumeration

impl Copy for [MyType] {}
//~^ ERROR the trait `Copy` may not be implemented for this type
//~| NOTE type is not a structure or enumeration
//~| ERROR only traits defined in the current crate can be implemented for arbitrary types
//~| NOTE impl doesn't use types inside crate
//~| NOTE the impl does not reference any types defined in this crate

impl Copy for &'static [NotSync] {}
//~^ ERROR the trait `Copy` may not be implemented for this type
//~| NOTE type is not a structure or enumeration
//~| ERROR only traits defined in the current crate can be implemented for arbitrary types
//~| NOTE impl doesn't use types inside crate
//~| NOTE the impl does not reference any types defined in this crate

fn main() {
}
