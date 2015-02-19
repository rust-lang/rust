// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(optin_builtin_traits)]

// @matches foo/struct.Alpha.html '//pre' "pub struct Alpha"
pub struct Alpha;
// @matches foo/struct.Bravo.html '//pre' "pub struct Bravo<B>"
pub struct Bravo<B>(B);

// @matches foo/struct.Alpha.html '//*[@class="impl"]//code' "impl !Send for Alpha"
impl !Send for Alpha {}

// @matches foo/struct.Bravo.html '//*[@class="impl"]//code' "impl<B> !Send for Bravo<B>"
impl<B> !Send for Bravo<B> {}
