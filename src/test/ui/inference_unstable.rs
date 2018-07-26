// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensures #[unstable] functions without opting in the corresponding #![feature]
// will not break inference.

// aux-build:inference_unstable_iterator.rs
// aux-build:inference_unstable_itertools.rs
// run-pass

extern crate inference_unstable_iterator;
extern crate inference_unstable_itertools;

#[allow(unused_imports)]
use inference_unstable_iterator::IpuIterator;
use inference_unstable_itertools::IpuItertools;

fn main() {
    assert_eq!('x'.ipu_flatten(), 1);
    //~^ WARN a method with this name may be added to the standard library in the future
    //~^^ WARN once this method is added to the standard library, the ambiguity may cause an error
}
