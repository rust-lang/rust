// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15679


#![feature(non_ascii_idents)]

use std::num;

pub fn main() {
    let ε = 0.00001f64;
    let Π = 3.14f64;
    let लंच = Π * Π + 1.54;
    assert!(num::abs((लंच - 1.54) - (Π * Π)) < ε);
    assert_eq!(საჭმელად_გემრიელი_სადილი(), 0);
}

fn საჭმელად_გემრიელი_სადილი() -> int {

    // Lunch in several languages.

    let ランチ = 10i;
    let 午餐 = 10i;

    let ארוחת_צהריי = 10i;
    let غداء = 10u;
    let լանչ = 10i;
    let обед = 10i;
    let абед = 10i;
    let μεσημεριανό = 10i;
    let hádegismatur = 10i;
    let ручек = 10i;

    let ăn_trưa = 10i;
    let อาหารกลางวัน = 10i;

    // Lunchy arithmetic, mm.

    assert_eq!(hádegismatur * ручек * обед, 1000);
    assert_eq!(10i, ארוחת_צהריי);
    assert_eq!(ランチ + 午餐 + μεσημεριανό, 30);
    assert_eq!(ăn_trưa + อาหารกลางวัน, 20);
    return (абед + լանչ) >> غداء;
}
