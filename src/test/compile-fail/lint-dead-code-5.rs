// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(struct_variant)]
#![allow(unused_variable)]
#![deny(dead_code)]

enum Enum1 {
    Variant1(int),
    Variant2 //~ ERROR: variant is never used
}

enum Enum2 {
    Variant3(bool),
    #[allow(dead_code)]
    Variant4(int),
    Variant5 { _x: int }, //~ ERROR: variant is never used: `Variant5`
    Variant6(int), //~ ERROR: variant is never used: `Variant6`
    _Variant7,
}

enum Enum3 { //~ ERROR: enum is never used
    Variant8,
    Variant9
}

fn main() {
    let v = Variant1(1);
    match v {
        Variant1(_) => (),
        Variant2 => ()
    }
    let x = Variant3(true);
}
