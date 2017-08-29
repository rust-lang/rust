// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:derive-b.rs
// ignore-stage1

#![allow(warnings)]

#[macro_use]
extern crate derive_b;

#[derive(B)]
#[B]
#[C] //~ ERROR: The attribute `C` is currently unknown to the compiler
#[B(D)]
#[B(E = "foo")]
#[B arbitrary tokens] //~ expected one of `(` or `=`, found `arbitrary`
struct B;

fn main() {}
