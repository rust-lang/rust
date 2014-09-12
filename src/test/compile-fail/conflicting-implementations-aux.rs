// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #3512 - conflicting trait impls in different crates should give a
// 'conflicting implementations' error message.

// aux-build:trait_impl_conflict.rs
extern crate trait_impl_conflict;
use trait_impl_conflict::Foo;

impl<A> Foo for A { //~ ERROR E0117
}

fn main() {
}
