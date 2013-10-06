// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(struct_variant)];

#[deriving(ToStr)]
enum A {}
#[deriving(ToStr)]
enum B { B1, B2, B3 }
#[deriving(ToStr)]
enum C { C1(int), C2(B), C3(~str) }
#[deriving(ToStr)]
enum D { D1{ a: int } }
#[deriving(ToStr)]
struct E;
#[deriving(ToStr)]
struct F(int);
#[deriving(ToStr)]
struct G(int, int);
#[deriving(ToStr)]
struct H { a: int }
#[deriving(ToStr)]
struct I { a: int, b: int }
#[deriving(ToStr)]
struct J(Custom);

struct Custom;
impl ToStr for Custom {
    fn to_str(&self) -> ~str { ~"yay" }
}

pub fn main() {
    assert_eq!(B1.to_str(), ~"B1");
    assert_eq!(B2.to_str(), ~"B2");
    assert_eq!(C1(3).to_str(), ~"C1(3)");
    assert_eq!(C2(B2).to_str(), ~"C2(B2)");
    assert_eq!(D1{ a: 2 }.to_str(), ~"D1{a: 2}");
    assert_eq!(E.to_str(), ~"E");
    assert_eq!(F(3).to_str(), ~"F(3)");
    assert_eq!(G(3, 4).to_str(), ~"G(3, 4)");
    assert_eq!(G(3, 4).to_str(), ~"G(3, 4)");
    assert_eq!(I{ a: 2, b: 4 }.to_str(), ~"I{a: 2, b: 4}");
    assert_eq!(J(Custom).to_str(), ~"J(yay)");
}
