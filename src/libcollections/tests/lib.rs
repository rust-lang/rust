// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(warnings)]

#![feature(binary_heap_peek_mut_pop)]
#![feature(box_syntax)]
#![feature(inclusive_range_syntax)]
#![feature(collection_placement)]
#![feature(collections)]
#![feature(const_fn)]
#![feature(exact_size_is_empty)]
#![feature(pattern)]
#![feature(placement_in_syntax)]
#![feature(rand)]
#![feature(splice)]
#![feature(step_by)]
#![feature(str_escape)]
#![feature(test)]
#![feature(unboxed_closures)]
#![feature(unicode)]
#![feature(utf8_error_error_len)]

extern crate collections;
extern crate test;
extern crate std_unicode;
extern crate core;

use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

mod binary_heap;
mod btree;
mod cow_str;
mod fmt;
mod linked_list;
mod slice;
mod str;
mod string;
mod vec_deque;
mod vec;

fn hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}
