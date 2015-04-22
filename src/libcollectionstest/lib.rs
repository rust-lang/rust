// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]
#![feature(collections)]
#![feature(core)]
#![feature(hash)]
#![feature(rand)]
#![feature(rustc_private)]
#![feature(test)]
#![feature(unboxed_closures)]
#![feature(unicode)]
#![feature(unsafe_destructor)]
#![feature(into_cow)]
#![feature(step_by)]
#![cfg_attr(test, feature(str_char))]

#[macro_use] extern crate log;

extern crate collections;
extern crate test;
extern crate rustc_unicode;

#[cfg(test)] #[macro_use] mod bench;

mod binary_heap;
mod bit;
mod btree;
mod enum_set;
mod fmt;
mod linked_list;
mod slice;
mod str;
mod string;
mod vec_deque;
mod vec_map;
mod vec;
