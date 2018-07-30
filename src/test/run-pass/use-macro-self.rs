// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:use-macro-self.rs

#![feature(use_extern_macros)]

#[macro_use]
extern crate use_macro_self;

use use_macro_self::foobarius::{self};

fn main() {
    let _: () = foobarius!(); // OK, the macro returns `()`
}
