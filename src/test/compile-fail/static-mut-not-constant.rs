// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

static mut a: Box<isize> = box 3;
//~^ ERROR mutable statics are not allowed to have owned pointers
//~| ERROR statics are not allowed to have destructors
//^| ERROR statics are not allowed to have destructors
//~| ERROR statics are not allowed to have destructors
//~| ERROR statics are not allowed to have destructors
//~| ERROR blocks in statics are limited to items and tail expressions
//~| ERROR blocks in statics are limited to items and tail expressions
//~| ERROR blocks in statics are limited to items and tail expressions
//~| ERROR blocks in statics are limited to items and tail expressions
//~| ERROR function calls in statics are limited to struct and enum constructors
//~| ERROR function calls in statics are limited to struct and enum constructors
//~| ERROR function calls in statics are limited to struct and enum constructors
//~| ERROR function calls in statics are limited to struct and enum constructors
//~| ERROR paths in statics may only refer to constants or functions
//~| ERROR paths in statics may only refer to constants or functions
//~| ERROR paths in statics may only refer to constants or functions
//~| ERROR paths in statics may only refer to constants or functions
//~| ERROR references in statics may only refer to immutable values

fn main() {}
