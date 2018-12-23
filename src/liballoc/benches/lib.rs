// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(repr_simd)]
#![feature(slice_sort_by_cached_key)]
#![feature(test)]

extern crate rand;
extern crate rand_xorshift;
extern crate test;

mod btree;
mod linked_list;
mod string;
mod str;
mod slice;
mod vec;
mod vec_deque;
