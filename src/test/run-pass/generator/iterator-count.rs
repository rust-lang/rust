// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generators, generator_trait, conservative_impl_trait)]

use std::ops::{State, Generator};

struct W<T>(T);

impl<T: Generator<Return = ()>> Iterator for W<T> {
    type Item = T::Yield;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.resume(()) {
            State::Complete(..) => None,
            State::Yielded(v) => Some(v),
        }
    }
}

fn test() -> impl Generator<Return=(), Yield=u8> {
    || {
        for i in 1..6 {
            yield i
        }
    }
}

fn main() {
    let end = 11;

    let closure_test = |start| {
        move || {
            for i in start..end {
                yield i
            }
        }
    };

    assert!(W(test()).chain(W(closure_test(6))).eq(1..11));
}
