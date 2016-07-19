// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![allow(unused)]

macro_rules! make_item {
    () => { fn f() {} }
}

macro_rules! make_stmt {
    () => { let x = 0; }
}

fn f() {
    make_item! {}
}

fn g() {
    make_stmt! {}
}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
