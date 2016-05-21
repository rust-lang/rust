// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// regression test for #32950

#![feature(type_macros)]

macro_rules! passthru {
    ($t:ty) => { $t }
}

macro_rules! useassoc {
    ($t:ident, $assoc:ident) => { ($t, $t::$assoc) }
}

trait HasAssoc { type Type; }

impl HasAssoc for i32 { type Type = i32; }

#[derive(Debug)]
struct Thing1<T: HasAssoc> {
    thing: useassoc!(T, Type)
}
#[derive(Debug)]
struct Thing2<T: HasAssoc> {
    thing: (T, T::Type)
}

fn main() {
    let t1: passthru!(Thing1<passthru!(i32)>) = Thing1 { thing: (42, 42) };
    let t2: passthru!(Thing2<passthru!(i32)>) = Thing2 { thing: (42, 42) };
    println!("{:?} {:?}", t1, t2);
}

