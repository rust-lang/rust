// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// `#[derive(Trait)]` works for empty structs/variants with braces or parens.

#![feature(rustc_private)]

extern crate serialize as rustc_serialize;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
         Default, Debug, RustcEncodable, RustcDecodable)]
struct S {}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
         Default, Debug, RustcEncodable, RustcDecodable)]
struct Z();

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash,
         Debug, RustcEncodable, RustcDecodable)]
enum E {
    V {},
    U,
    W(),
}

fn main() {
    let s = S {};
    let s1 = s;
    let s2 = s.clone();
    assert_eq!(s, s1);
    assert_eq!(s, s2);
    assert!(!(s < s1));
    assert_eq!(format!("{:?}", s), "S");

    let z = Z();
    let z1 = z;
    let z2 = z.clone();
    assert_eq!(z, z1);
    assert_eq!(z, z2);
    assert!(!(z < z1));
    assert_eq!(format!("{:?}", z), "Z");

    let e = E::V {};
    let e1 = e;
    let e2 = e.clone();
    assert_eq!(e, e1);
    assert_eq!(e, e2);
    assert!(!(e < e1));
    assert_eq!(format!("{:?}", e), "V");

    let e = E::W();
    let e1 = e;
    let e2 = e.clone();
    assert_eq!(e, e1);
    assert_eq!(e, e2);
    assert!(!(e < e1));
    assert_eq!(format!("{:?}", e), "W");
}
