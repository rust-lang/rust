// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Do not remove on snapshot creation. Needed for bootstrap. (Issue #22364)
#![cfg_attr(stage0, feature(custom_attribute))]
#![feature(box_syntax)]
#![feature(int_uint)]
#![feature(unboxed_closures)]
#![feature(unsafe_destructor)]
#![feature(core)]
#![feature(test)]
#![feature(rand)]
#![feature(unicode)]
#![feature(std_misc)]
#![feature(libc)]
#![feature(hash)]
#![feature(io)]
#![feature(collections)]
#![feature(debug_builders)]
#![feature(unique)]
#![feature(step_by)]
#![allow(deprecated)] // rand

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
