// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type="lib"]

// @has assoc_types/trait.Index.html
pub trait Index<I: ?Sized> {
    // @has - '//*[@id="associatedtype.Output"]//code' 'type Output: ?Sized'
    type Output: ?Sized;
    // @has - '//*[@id="tymethod.index"]//code' \
    //      "fn index<'a>(&'a self, index: I) -> &'a Self::Output"
    fn index<'a>(&'a self, index: I) -> &'a Self::Output;
}
