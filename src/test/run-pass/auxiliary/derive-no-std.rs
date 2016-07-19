// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic

#![crate_type = "rlib"]
#![no_std]

// Issue #16803

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord,
         Debug, Default, Copy)]
pub struct Foo {
    pub x: u32,
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord,
         Debug, Copy)]
pub enum Bar {
    Qux,
    Quux(u32),
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord,
         Debug, Copy)]
pub enum Void {}
#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord,
         Debug, Copy)]
pub struct Empty;
#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord,
         Debug, Copy)]
pub struct AlsoEmpty {}

