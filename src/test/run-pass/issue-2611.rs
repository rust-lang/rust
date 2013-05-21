// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::old_iter::BaseIter;

trait FlatMapToVec<A> {
  fn flat_map_to_vec<B, IB:BaseIter<B>>(&self, op: &fn(&A) -> IB) -> ~[B];
}

impl<A:Copy> FlatMapToVec<A> for ~[A] {
   fn flat_map_to_vec<B, IB:BaseIter<B>>(&self, op: &fn(&A) -> IB) -> ~[B] {
     old_iter::flat_map_to_vec(self, op)
   }
}

pub fn main() {}
