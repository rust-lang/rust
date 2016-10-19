// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check for const_err regressions

#![deny(const_err)]

const X: *const u8 = b"" as _;

fn main() {
    let _ = ((-1 as i8) << 8 - 1) as f32;
    let _ = 0u8 as char;
    let _ = true > false;
    let _ = true >= false;
    let _ = true < false;
    let _ = true >= false;
}
