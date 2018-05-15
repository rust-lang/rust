// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue_50493.rs
// ignore-stage1

#![feature(proc_macro)]

#[macro_use]
extern crate issue_50493;

#[derive(Derive)] //~ ERROR field `field` of struct `Restricted` is private
struct Restricted {
    pub(in restricted) field: usize, //~ visibilities can only be restricted to ancestor modules
}

mod restricted {}

fn main() {}

