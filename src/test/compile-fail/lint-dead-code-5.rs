// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_variables)]
#![deny(dead_code)]

enum Enum1 {
    Variant1(isize),
    Variant2 //~ ERROR: variant is never constructed
}

enum Enum2 {
    Variant3(bool),
    #[allow(dead_code)]
    Variant4(isize),
    Variant5 { _x: isize }, //~ ERROR: variant is never constructed: `Variant5`
    Variant6(isize), //~ ERROR: variant is never constructed: `Variant6`
    _Variant7,
}

enum Enum3 { //~ ERROR: enum is never used
    Variant8,
    Variant9
}

fn main() {
    let v = Enum1::Variant1(1);
    match v {
        Enum1::Variant1(_) => (),
        Enum1::Variant2 => ()
    }
    let x = Enum2::Variant3(true);
}
