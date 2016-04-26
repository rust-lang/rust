// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "rlib"]

#![allow(unused_variables)]
#![feature(omit_gdb_pretty_printer_section)]
#![omit_gdb_pretty_printer_section]

// no-prefer-dynamic
// compile-flags:-g

pub fn generic_function<T: Clone>(val: T) -> (T, T) {
    let result = (val.clone(), val.clone());
    let a_variable: u32 = 123456789;
    let another_variable: f64 = 123456789.5;
    zzz();
    result
}

#[inline(never)]
fn zzz() {()}
