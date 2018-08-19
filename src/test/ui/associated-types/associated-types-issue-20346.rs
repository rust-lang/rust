// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we reliably check the value of the associated type.

#![crate_type = "lib"]
#![no_implicit_prelude]

use std::option::Option::{self, None, Some};
use std::vec::Vec;

trait Iterator {
    type Item;

    fn next(&mut self) -> Option<Self::Item>;
}

fn is_iterator_of<A, I: Iterator<Item=A>>(_: &I) {}

struct Adapter<I> {
    iter: I,
    found_none: bool,
}

impl<T, I> Iterator for Adapter<I> where I: Iterator<Item=Option<T>> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        loop {}
    }
}

fn test_adapter<T, I: Iterator<Item=Option<T>>>(it: I) {
    is_iterator_of::<Option<T>, _>(&it);  // Sanity check
    let adapter = Adapter { iter: it, found_none: false };
    is_iterator_of::<T, _>(&adapter); // OK
    is_iterator_of::<Option<T>, _>(&adapter); //~ ERROR type mismatch
}
