// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(match_default_bindings)]

// FIXME(tschottdorf): This should compile. See #44912.

pub fn main() {
    let x = &Some((3, 3));
    let _: &i32 = match x {
        Some((x, 3)) | &Some((ref x, 5)) => x,
        //~^ ERROR is bound in inconsistent ways
        _ => &5i32,
    };
}
