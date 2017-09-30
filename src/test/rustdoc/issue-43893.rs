// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-cross-compile

#![crate_name = "foo"]

pub trait SomeTrait {}

pub struct SomeStruct;

//

// dummy
// @has foo/trait.SomeTrait.html '//a/@href' '../src/foo/issue-43893.rs.html#23-26'
impl SomeTrait for usize {


}

// @has foo/trait.SomeTrait.html '//a/@href' '../src/foo/issue-43893.rs.html#29-32'
impl SomeTrait for SomeStruct {


}

// some trailer
