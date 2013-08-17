// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::uint;

pub fn main() {
    // sometimes we have had trouble finding
    // the right type for f, as we unified
    // bot and u32 here
    let f = match uint::from_str("1234") {
        None => return (),
        Some(num) => num as u32
    };
    assert_eq!(f, 1234u32);
    error!(f)
}
