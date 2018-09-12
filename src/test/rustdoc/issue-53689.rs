// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-53689.rs

#![crate_name = "foo"]

extern crate issue_53689;

// @has foo/trait.MyTrait.html
// @!has - 'MyStruct'
// @count - '//*[code="impl<T> MyTrait for T"]' 1
pub trait MyTrait {}

impl<T> MyTrait for T {}

mod a {
    pub use issue_53689::MyStruct;
}
