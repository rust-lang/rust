// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// If the unstable API is the only possible solution,
// still emit E0658 "use of unstable library feature".

// aux-build:inference_unstable_iterator.rs

extern crate inference_unstable_iterator;

use inference_unstable_iterator::IpuIterator;

fn main() {
    assert_eq!('x'.ipu_flatten(), 0);   //~ ERROR E0658
}
