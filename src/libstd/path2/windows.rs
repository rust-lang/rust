// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Windows file path handling

/// The standard path separator character
pub static sep: u8 = '\\' as u8;
/// The alternative path separator character
pub static sep2: u8 = '/' as u8;

/// Returns whether the given byte is a path separator
#[inline]
pub fn is_sep(u: &u8) -> bool {
    *u == sep || *u == sep2
}
