// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

use std::fmt::Debug;
use std::default::Default;

// Test that an impl for homogeneous pairs does not conflict with a
// heterogeneous pair.

trait MyTrait {
    fn get(&self) -> usize;
}

impl<T> MyTrait for (T,T) {
    fn get(&self) -> usize { 0 }
}

impl MyTrait for (usize,isize) {
    fn get(&self) -> usize { 0 }
}

fn main() {
}
