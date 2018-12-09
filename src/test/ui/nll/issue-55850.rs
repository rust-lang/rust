// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_mut)]
#![feature(generators, generator_trait)]

use std::ops::Generator;
use std::ops::GeneratorState::Yielded;

pub struct GenIter<G>(G);

impl <G> Iterator for GenIter<G>
where
    G: Generator,
{
    type Item = G::Yield;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            match self.0.resume() {
                Yielded(y) => Some(y),
                _ => None
            }
        }
    }
}

fn bug<'a>() -> impl Iterator<Item = &'a str> {
    GenIter(move || {
        let mut s = String::new();
        yield &s[..] //~ ERROR `s` does not live long enough [E0597]
    })
}

fn main() {
    bug();
}
