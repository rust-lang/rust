// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-wasm32-bare compiled with panic=abort by default

#![feature(generators, generator_trait)]

use std::ops::{GeneratorState, Generator};
use std::panic;

fn main() {
    let mut foo = || {
        if true {
            return
        }
        yield;
    };

    match unsafe { foo.resume() } {
        GeneratorState::Complete(()) => {}
        s => panic!("bad state: {:?}", s),
    }

    match panic::catch_unwind(move || unsafe { foo.resume() }) {
        Ok(_) => panic!("generator successfully resumed"),
        Err(_) => {}
    }
}
