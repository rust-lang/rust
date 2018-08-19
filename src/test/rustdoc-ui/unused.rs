// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

// This test purpose is to check that unused_imports lint isn't fired
// by rustdoc. Why would it? Because when rustdoc is running, it uses
// "everybody-loops" which replaces parts of code with "loop {}" to get
// huge performance improvements.

#![deny(unused_imports)]

use std::fs::File;

pub fn f() {
    let _: File;
}
