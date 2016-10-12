// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:custom_derive_partial_eq.rs
// ignore-stage1
#![feature(plugin, custom_derive)]
#![plugin(custom_derive_partial_eq)]
#![allow(unused)]

#[derive(CustomPartialEq)] // Check that this is not a stability error.
enum E { V1, V2 }

fn main() {}
