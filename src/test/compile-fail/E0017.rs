// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static X: i32 = 1;
const C: i32 = 2;

const CR: &'static mut i32 = &mut C; //~ ERROR E0017
                                     //~| ERROR E0017
static STATIC_REF: &'static mut i32 = &mut X; //~ ERROR E0017
                                              //~| ERROR E0017
                                              //~| ERROR E0388
static CONST_REF: &'static mut i32 = &mut C; //~ ERROR E0017
                                             //~| ERROR E0017

fn main() {}
