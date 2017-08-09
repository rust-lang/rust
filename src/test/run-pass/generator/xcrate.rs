// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:xcrate.rs

#![feature(generators, generator_trait)]

extern crate xcrate;

use std::ops::{GeneratorState, Generator};

fn main() {
    let mut foo = xcrate::foo();

    match foo.resume() {
        GeneratorState::Complete(()) => {}
        s => panic!("bad state: {:?}", s),
    }
}
