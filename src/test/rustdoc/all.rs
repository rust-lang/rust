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

// @has foo/all.html '//a[@href="struct.Struct.html"]' 'Struct'
// @has foo/all.html '//a[@href="enum.Enum.html"]' 'Enum'
// @has foo/all.html '//a[@href="union.Union.html"]' 'Union'
// @has foo/all.html '//a[@href="constant.CONST.html"]' 'CONST'
// @has foo/all.html '//a[@href="static.STATIC.html"]' 'STATIC'
// @has foo/all.html '//a[@href="fn.function.html"]' 'function'

pub struct Struct;
pub enum Enum {
    X,
    Y,
}
pub union Union {
    x: u32,
}
pub const CONST: u32 = 0;
pub static STATIC: &str = "baguette";
pub fn function() {}

mod private_module {
    pub struct ReexportedStruct;
}

// @has foo/all.html '//a[@href="struct.ReexportedStruct.html"]' 'ReexportedStruct'
// @!has foo/all.html 'private_module'
pub use private_module::ReexportedStruct;
