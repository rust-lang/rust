// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty `dyn ::foo` parses differently in the current edition

use std::fmt::Display;

static BYTE: u8 = 33;

fn main() {
    let x: &(dyn 'static + Display) = &BYTE;
    let y: Box<dyn Display + 'static> = Box::new(BYTE);
    let _: &dyn (Display) = &BYTE;
    let _: &dyn (::std::fmt::Display) = &BYTE;
    let xstr = format!("{}", x);
    let ystr = format!("{}", y);
    assert_eq!(xstr, "33");
    assert_eq!(ystr, "33");
}
