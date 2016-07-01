// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(warnings)]

#![feature(as_unsafe_cell)]
#![feature(borrow_state)]
#![feature(box_syntax)]
#![feature(cell_extras)]
#![feature(const_fn)]
#![feature(core_private_bignum)]
#![feature(core_private_diy_float)]
#![feature(dec2flt)]
#![feature(fixed_size_array)]
#![feature(float_extras)]
#![feature(flt2dec)]
#![feature(iter_arith)]
#![feature(libc)]
#![feature(nonzero)]
#![feature(rand)]
#![feature(raw)]
#![feature(sip_hash_13)]
#![feature(slice_patterns)]
#![feature(step_by)]
#![feature(test)]
#![feature(unboxed_closures)]
#![feature(unicode)]
#![feature(unique)]
#![feature(try_from)]

extern crate core;
extern crate test;
extern crate libc;
extern crate rustc_unicode;
extern crate rand;

mod any;
mod array;
mod atomic;
mod cell;
mod char;
mod clone;
mod cmp;
mod fmt;
mod hash;
mod intrinsics;
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
