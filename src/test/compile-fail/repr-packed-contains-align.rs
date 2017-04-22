// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(attr_literals)]
#![feature(repr_align)]
#![allow(dead_code)]

#[repr(align(16))]
struct A(i32);

struct B(A);

#[repr(packed)]
struct C(A); //~ ERROR: packed struct cannot transitively contain a `[repr(align)]` struct

#[repr(packed)]
struct D(B); //~ ERROR: packed struct cannot transitively contain a `[repr(align)]` struct

fn main() {}
