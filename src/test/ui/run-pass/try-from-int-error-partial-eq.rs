// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass

#![feature(try_from)]
#![allow(unused_must_use)]

use std::convert::TryFrom;
use std::num::TryFromIntError;

fn main() {
    let x: u32 = 125;
    let y: Result<u8, TryFromIntError> = u8::try_from(x);
    y == Ok(125);
}
