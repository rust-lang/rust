// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(as_unsafe_cell)]
#![feature(borrow_state)]
#![feature(box_syntax)]
#![feature(cell_extras)]
#![feature(const_fn)]
#![feature(core)]
#![feature(core_float)]
#![feature(float_extras)]
#![feature(float_from_str_radix)]
#![feature(flt2dec)]
#![feature(dec2flt)]
#![feature(fmt_radix)]
#![feature(iter_arith)]
#![feature(iter_arith)]
#![feature(iter_cmp)]
#![feature(iter_order)]
#![feature(libc)]
#![feature(nonzero)]
#![feature(num_bits_bytes)]
#![feature(ptr_as_ref)]
#![feature(rand)]
#![feature(range_inclusive)]
#![feature(raw)]
#![feature(result_expect)]
#![feature(slice_bytes)]
#![feature(slice_patterns)]
#![feature(step_by)]
#![feature(test)]
#![feature(unboxed_closures)]
#![feature(unicode)]
#![feature(unique)]

extern crate core;
extern crate test;
extern crate libc;
extern crate rustc_unicode;
extern crate rand;

mod any;
mod atomic;
mod cell;
mod char;
mod cmp;
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
