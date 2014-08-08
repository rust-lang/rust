// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::ascii::{Ascii, AsciiCast};

mod num;

#[test]
fn test_to_string() {
    let s = b't'.to_ascii().to_string();
    assert_eq!(s, "t".to_string());
}

#[test]
fn test_show() {
    let c = b't'.to_ascii();
    assert_eq!(format!("{}", c), "t".to_string());
}
