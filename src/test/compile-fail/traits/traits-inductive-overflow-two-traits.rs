// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #29859, initial version. This example allowed
// arbitrary trait bounds to be synthesized.

// Trait that you want all types to implement.
use std::marker::{Sync as Trait};

pub trait Magic {
    type X: Trait;
}
impl<T: Magic> Magic for T {
    type X = Self;
}

fn check<T: Trait>() {}

fn wizard<T: Magic>() { check::<<T as Magic>::X>(); }

fn main() {
    wizard::<*mut ()>(); //~ ERROR E0275
    // check::<*mut ()>();
}
