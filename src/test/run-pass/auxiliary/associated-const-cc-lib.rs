// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_consts)]

#![crate_type="lib"]

// These items are for testing that associated consts work cross-crate.
pub trait Foo {
    const BAR: usize;
}

pub struct FooNoDefault;

impl Foo for FooNoDefault {
    const BAR: usize = 0;
}

// These test that defaults and default resolution work cross-crate.
pub trait FooDefault {
    const BAR: usize = 1;
}

pub struct FooOverwriteDefault;

impl FooDefault for FooOverwriteDefault {
    const BAR: usize = 2;
}

pub struct FooUseDefault;

impl FooDefault for FooUseDefault {}

// Test inherent impls.
pub struct InherentBar;

impl InherentBar {
    pub const BAR: usize = 3;
}
