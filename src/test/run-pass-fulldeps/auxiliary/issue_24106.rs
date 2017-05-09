// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type="lib"]

enum E { E0 = 0, E1 = 1 }
const E0_U8: u8 = E::E0 as u8;
const E1_U8: u8 = E::E1 as u8;

pub fn go<T>() {
    match 0 {
        E0_U8 => (),
        E1_U8 => (),
        _ => (),
    }
}
