// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(ascii)]
#![feature(append)]
#![feature(bitset)]
#![feature(bitvec)]
#![feature(box_syntax)]
#![feature(btree_range)]
#![feature(collections)]
#![feature(collections_bound)]
#![feature(const_fn)]
#![feature(core)]
#![feature(deque_extras)]
#![feature(drain)]
#![feature(enumset)]
#![feature(hash_default)]
#![feature(into_cow)]
#![feature(iter_idx)]
#![feature(iter_order)]
#![feature(iter_arith)]
#![feature(iter_to_vec)]
#![feature(map_in_place)]
#![feature(move_from)]
#![feature(num_bits_bytes)]
#![feature(pattern)]
#![feature(permutations)]
#![feature(rand)]
#![feature(range_inclusive)]
#![feature(rustc_private)]
#![feature(slice_bytes)]
#![feature(slice_chars)]
#![feature(slice_splits)]
#![feature(slice_position_elem)]
#![feature(split_off)]
#![feature(step_by)]
#![feature(str_char)]
#![feature(str_escape)]
#![feature(str_match_indices)]
#![feature(str_split_at)]
#![feature(str_utf16)]
#![feature(box_str)]
#![feature(subslice_offset)]
#![feature(test)]
#![feature(unboxed_closures)]
#![feature(unicode)]
#![feature(vec_deque_retain)]
#![feature(vec_from_raw_buf)]
#![feature(vec_push_all)]
#![feature(vecmap)]

#![allow(deprecated)]

#[macro_use] extern crate log;

extern crate collections;
extern crate test;
extern crate rustc_unicode;

#[cfg(test)] #[macro_use] mod bench;

mod binary_heap;
mod bit;
mod btree;
mod fmt;
mod linked_list;
mod slice;
mod str;
mod string;
mod vec_deque;
mod vec_map;
mod vec;
