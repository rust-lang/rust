// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

//[rpass1] compile-flags: -g
//[rpass2] compile-flags: -g
//[rpass3] compile-flags: -g --remap-path-prefix={{src-base}}=/the/src

#![feature(rustc_attrs)]
#![crate_type="rlib"]

#[inline(always)]
pub fn inline_fn() {
    println!("test");
}
