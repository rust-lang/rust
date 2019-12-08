// Copyright 2014-2019 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-rustfix
// aux-build:macro_rules.rs
#![allow(unused_imports)]
#![warn(
    clippy::all,
    clippy::style,
    clippy::mem_replace_option_with_none,
    clippy::mem_replace_with_default
)]

#[macro_use]
extern crate macro_rules;

use std::mem;

macro_rules! take {
    ($s:expr) => {
        std::mem::replace($s, Default::default())
    };
}

fn replace_option_with_none() {
    let mut an_option = Some(1);
    let _ = mem::replace(&mut an_option, None);
    let an_option = &mut Some(1);
    let _ = mem::replace(an_option, None);
}

fn replace_with_default() {
    let mut s = String::from("foo");
    let _ = std::mem::replace(&mut s, String::default());
    let s = &mut String::from("foo");
    let _ = std::mem::replace(s, String::default());
    let _ = std::mem::replace(s, Default::default());

    // dont lint within macros
    take!(s);
    take_external!(s);
}

fn main() {
    replace_option_with_none();
    replace_with_default();
}
