// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(pub_restricted)]

macro_rules! m {
    ($p: path) => (pub($p) struct Z;)
}

struct S<T>(T);
m!{ S<u8> } //~ ERROR type or lifetime parameters in visibility path
//~^ ERROR expected module, found struct `S`

mod foo {
    struct S(pub(foo<T>) ()); //~ ERROR type or lifetime parameters in visibility path
    //~^ ERROR cannot find type `T` in this scope
}

fn main() {}
