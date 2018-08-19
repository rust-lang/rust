// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: a b c

// aux-build:coherence_lib.rs

// Test that the `Pair` type reports an error if it contains type
// parameters, even when they are covered by local types. This test
// was originally intended to test the opposite, but the rules changed
// with RFC 1023 and this became illegal.

extern crate coherence_lib as lib;
use lib::{Remote,Pair};

pub struct Cover<T>(T);

#[cfg(a)]
impl<T> Remote for Pair<T,Cover<T>> { } //[a]~ ERROR E0210

#[cfg(b)]
impl<T> Remote for Pair<Cover<T>,T> { } //[b]~ ERROR E0210

#[cfg(c)]
impl<T,U> Remote for Pair<Cover<T>,U> { }
//[c]~^ ERROR type parameter `T` must be used as the type parameter for some local type

fn main() { }
