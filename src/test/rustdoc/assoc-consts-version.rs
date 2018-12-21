// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

#![crate_name = "foo"]

#![feature(staged_api)]

#![stable(since="1.1.1", feature="rust1")]

#[stable(since="1.1.1", feature="rust1")]
pub struct SomeStruct;

impl SomeStruct {
    // @has 'foo/struct.SomeStruct.html' '//*[@id="SOME_CONST.v"]//div[@class="since"]' '1.1.2'
    #[stable(since="1.1.2", feature="rust2")]
    pub const SOME_CONST: usize = 0;
}
