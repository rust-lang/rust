// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem::transmute;

fn main() {
    unsafe {
        let _: i8 = transmute(false);
        let _: i8 = transmute(true);
        let _: bool = transmute(0u8);
        let _: bool = transmute(1u8);
    }
}
