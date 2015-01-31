// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]
#![feature(int_uint)]
#![feature(unboxed_closures)]
#![feature(unsafe_destructor, slicing_syntax)]

extern crate core;
extern crate test;
extern crate libc;
extern crate unicode;

mod any;
mod atomic;
mod cell;
mod char;
mod cmp;
mod finally;
mod fmt;
mod hash;
mod iter;
mod mem;
mod nonzero;
mod num;
mod ops;
mod option;
mod ptr;
mod result;
mod slice;
mod str;
mod tuple;
