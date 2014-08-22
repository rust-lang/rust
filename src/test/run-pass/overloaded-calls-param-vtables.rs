// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that nested vtables work with overloaded calls.

#![feature(overloaded_calls)]

use std::ops::Fn;

struct G;

impl<'a, A: Add<int, int>> Fn<(A,), int> for G {
    extern "rust-call" fn call(&self, (arg,): (A,)) -> int {
        arg.add(&1)
    }
}

fn main() {
    // ICE trigger
    G(1i);
}

