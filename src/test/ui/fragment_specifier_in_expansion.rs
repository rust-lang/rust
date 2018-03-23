// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests the "fragment specifier in expansion" warning

// must-compile-successfully

#![allow(unused)]

macro_rules! warn_me {
    ( $thing:tt ) => { $thing:tt }
    //~^ WARN fragment specifier
}

macro_rules! with_space {
    ( $thing:tt ) => { $thing: tt }
}

macro_rules! unknown_specifier {
    ( $thing:tt ) => { $thing:foobar }
}

fn main() {}
