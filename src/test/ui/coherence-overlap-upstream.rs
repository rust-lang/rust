// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that we consider `i16: Remote` to be ambiguous, even
// though the upstream crate doesn't implement it for now.

// aux-build:coherence_lib.rs

extern crate coherence_lib;

use coherence_lib::Remote;

trait Foo {}
impl<T> Foo for T where T: Remote {}
impl Foo for i16 {}
//~^ ERROR E0119

fn main() {}
