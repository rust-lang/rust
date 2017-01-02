// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:derive-atob.rs
// aux-build:derive-ctod.rs

#[macro_use]
extern crate derive_atob;
#[macro_use]
extern crate derive_ctod;

#[derive(Copy, Clone)]
#[derive(AToB)]
struct A;

#[derive(CToD)]
struct C;

fn main() {
    B;
    D;
}
