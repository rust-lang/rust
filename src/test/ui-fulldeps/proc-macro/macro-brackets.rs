// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:macro-brackets.rs

#![feature(proc_macro)]

extern crate macro_brackets as bar;
use bar::doit;

macro_rules! id {
    ($($t:tt)*) => ($($t)*)
}

#[doit]
id![static X: u32 = 'a';]; //~ ERROR: mismatched types


fn main() {}
