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

#![feature(box_syntax)]
#![feature(char_escape_debug)]
#![feature(const_fn)]
#![feature(core_float)]
#![feature(core_private_bignum)]
#![feature(core_private_diy_float)]
#![feature(dec2flt)]
#![feature(decode_utf8)]
#![feature(fixed_size_array)]
#![feature(flt2dec)]
#![feature(fmt_internals)]
#![feature(iterator_step_by)]
#![feature(i128_type)]
#![feature(inclusive_range)]
#![feature(iter_rfind)]
#![feature(libc)]
#![feature(nonzero)]
#![feature(ord_max_min)]
#![feature(rand)]
#![feature(raw)]
#![feature(sip_hash_13)]
#![feature(slice_patterns)]
#![feature(slice_rotate)]
#![feature(sort_internals)]
#![feature(specialization)]
#![feature(step_trait)]
#![feature(test)]
#![feature(trusted_len)]
#![feature(try_from)]
#![feature(unicode)]
#![feature(unique)]

extern crate core;
extern crate test;
extern crate libc;
extern crate std_unicode;
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
