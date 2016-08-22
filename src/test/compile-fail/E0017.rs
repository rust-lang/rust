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
                                     //~| NOTE constants require immutable values
                                     //~| ERROR E0017
                                     //~| NOTE constants require immutable values
static STATIC_REF: &'static mut i32 = &mut X; //~ ERROR E0017
                                              //~| NOTE statics require immutable values
                                              //~| ERROR E0017
                                              //~| NOTE statics require immutable values
                                              //~| ERROR E0388
                                              //~| NOTE cannot write data in a static definition
static CONST_REF: &'static mut i32 = &mut C; //~ ERROR E0017
                                             //~| NOTE statics require immutable values
                                             //~| ERROR E0017
                                             //~| NOTE statics require immutable values
fn main() {}
