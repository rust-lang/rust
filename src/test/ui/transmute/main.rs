// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// normalize-stderr-32bit: "&str \(64 bits\)" -> "&str ($$STR bits)"
// normalize-stderr-64bit: "&str \(128 bits\)" -> "&str ($$STR bits)"



#![feature(untagged_unions)]
use std::mem::transmute;

pub trait TypeConstructor<'a> {
    type T;
}

unsafe fn transmute_lifetime<'a, 'b, C>(x: <C as TypeConstructor<'a>>::T)
                                        -> <C as TypeConstructor<'b>>::T
where for<'z> C: TypeConstructor<'z> {
    transmute(x) //~ ERROR transmute called with types of different sizes
}

unsafe fn sizes() {
    let x: u8 = transmute(10u16); //~ ERROR transmute called with types of different sizes
}

unsafe fn ptrs() {
    let x: u8 = transmute("test"); //~ ERROR transmute called with types of different sizes
}

union Foo { x: () }
unsafe fn vary() {
    let x: Foo = transmute(10); //~ ERROR transmute called with types of different sizes
}

fn main() {}
