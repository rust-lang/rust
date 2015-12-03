// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rand, collections, rustc_private)]
#![no_std]

extern crate rand;
extern crate serialize as rustc_serialize;
extern crate collections;

// Issue #16803

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord,
         Debug, Default, Copy)]
struct Foo {
    x: u32,
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord,
         Debug, Copy)]
enum Bar {
    Qux,
    Quux(u32),
}

enum Baz { A=0, B=5, }

fn main() {
    Foo { x: 0 };
    Bar::Quux(3);
    Baz::A;
}
