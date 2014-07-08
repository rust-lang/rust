// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(struct_variant)]

use std::fmt;

#[deriving(Show)]
enum A {}
#[deriving(Show)]
enum B { B1, B2, B3 }
#[deriving(Show)]
enum C { C1(int), C2(B), C3(String) }
#[deriving(Show)]
enum D { D1{ a: int } }
#[deriving(Show)]
struct E;
#[deriving(Show)]
struct F(int);
#[deriving(Show)]
struct G(int, int);
#[deriving(Show)]
struct H { a: int }
#[deriving(Show)]
struct I { a: int, b: int }
#[deriving(Show)]
struct J(Custom);

struct Custom;
impl fmt::Show for Custom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "yay")
    }
}

pub fn main() {
    assert_eq!(B1.to_string(), "B1".to_string());
    assert_eq!(B2.to_string(), "B2".to_string());
    assert_eq!(C1(3).to_string(), "C1(3)".to_string());
    assert_eq!(C2(B2).to_string(), "C2(B2)".to_string());
    assert_eq!(D1{ a: 2 }.to_string(), "D1 { a: 2 }".to_string());
    assert_eq!(E.to_string(), "E".to_string());
    assert_eq!(F(3).to_string(), "F(3)".to_string());
    assert_eq!(G(3, 4).to_string(), "G(3, 4)".to_string());
    assert_eq!(G(3, 4).to_string(), "G(3, 4)".to_string());
    assert_eq!(I{ a: 2, b: 4 }.to_string(), "I { a: 2, b: 4 }".to_string());
    assert_eq!(J(Custom).to_string(), "J(yay)".to_string());
}
