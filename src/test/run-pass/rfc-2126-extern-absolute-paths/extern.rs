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

#![feature(extern_in_paths)]

use extern::xcrate::Z;

type A = extern::xcrate::S;
type B = for<'a> extern::xcrate::Tr<'a>;

fn f() {
    use extern::xcrate;
    use extern::xcrate as ycrate;
    let s = xcrate::S;
    assert_eq!(format!("{:?}", s), "S");
    let z = ycrate::Z;
    assert_eq!(format!("{:?}", z), "Z");
}

fn main() {
    let s = extern::xcrate::S;
    assert_eq!(format!("{:?}", s), "S");
    let z = Z;
    assert_eq!(format!("{:?}", z), "Z");
    assert_eq!(A {}, extern::xcrate::S {});
}
