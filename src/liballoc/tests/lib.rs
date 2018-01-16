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

#![feature(allocator_api)]
#![feature(alloc_system)]
#![feature(attr_literals)]
#![feature(box_syntax)]
#![feature(inclusive_range_syntax)]
#![feature(collection_placement)]
#![feature(const_fn)]
#![feature(drain_filter)]
#![feature(exact_size_is_empty)]
#![feature(iterator_step_by)]
#![feature(pattern)]
#![feature(placement_in_syntax)]
#![feature(rand)]
#![feature(repr_align)]
#![feature(slice_rotate)]
#![feature(splice)]
#![feature(str_escape)]
#![feature(string_retain)]
#![feature(unboxed_closures)]
#![feature(unicode)]
#![feature(exact_chunks)]

extern crate alloc_system;
extern crate std_unicode;
extern crate rand;

use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

mod binary_heap;
mod btree;
mod cow_str;
mod fmt;
mod heap;
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

// FIXME: Instantiated functions with i128 in the signature is not supported in Emscripten.
// See https://github.com/kripken/emscripten-fastcomp/issues/169
#[cfg(not(target_os = "emscripten"))]
#[test]
fn test_boxed_hasher() {
    let ordinary_hash = hash(&5u32);

    let mut hasher_1 = Box::new(DefaultHasher::new());
    5u32.hash(&mut hasher_1);
    assert_eq!(ordinary_hash, hasher_1.finish());

    let mut hasher_2 = Box::new(DefaultHasher::new()) as Box<Hasher>;
    5u32.hash(&mut hasher_2);
    assert_eq!(ordinary_hash, hasher_2.finish());
}
