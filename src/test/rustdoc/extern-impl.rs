// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

// @has foo/struct.Foo.html
pub struct Foo;

impl Foo {
    // @has - '//code' 'fn rust0()'
    pub fn rust0() {}
    // @has - '//code' 'fn rust1()'
    pub extern "Rust" fn rust1() {}
    // @has - '//code' 'extern fn c0()'
    pub extern fn c0() {}
    // @has - '//code' 'extern fn c1()'
    pub extern "C" fn c1() {}
    // @has - '//code' 'extern "system" fn system0()'
    pub extern "system" fn system0() {}
}

// @has foo/trait.Bar.html
pub trait Bar {}

// @has - '//code' 'impl Bar for fn()'
impl Bar for fn() {}
// @has - '//code' 'impl Bar for extern fn()'
impl Bar for extern fn() {}
// @has - '//code' 'impl Bar for extern "system" fn()'
impl Bar for extern "system" fn() {}
