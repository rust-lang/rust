// Copyright 2012-2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Methods for dealing with boxed strings.
use core::mem;

use boxed::Box;

/// Converts a boxed slice of bytes to a boxed string slice without checking
/// that the string contains valid UTF-8.
#[unstable(feature = "str_box_extras", issue = "41119")]
pub unsafe fn from_boxed_utf8_unchecked(v: Box<[u8]>) -> Box<str> {
    mem::transmute(v)
}
