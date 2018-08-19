// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

// @has foo/index.html '//a[@href="../foo/foo/constant.FIRSTCONST.html"]' 'foo::FIRSTCONST'
// @has foo/index.html '//a[@href="../foo/struct.Bar.html#associatedconstant.CONST"]' 'Bar::CONST'

//! We have here [`foo::FIRSTCONST`] and [`Bar::CONST`].

pub mod foo {
    pub const FIRSTCONST: u32 = 42;
}

pub struct Bar;

impl Bar {
    pub const CONST: u32 = 42;
}
