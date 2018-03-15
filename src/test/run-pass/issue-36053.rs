// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #36053. ICE was caused due to obligations being
// added to a special, dedicated fulfillment cx during a
// probe. Problem seems to be related to the particular definition of
// `FusedIterator` in std but I was not able to isolate that into an
// external crate.

#![feature(fused)]
use std::iter::FusedIterator;

struct Thing<'a>(&'a str);
impl<'a> Iterator for Thing<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<&'a str> {
        None
    }
}

impl<'a> FusedIterator for Thing<'a> {}

fn main() {
    Thing("test").fuse().filter(|_| true).count();
}
