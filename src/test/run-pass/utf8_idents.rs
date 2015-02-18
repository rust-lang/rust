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

use std::num::Float;

pub fn main() {
    let ε = 0.00001f64;
    let Π = 3.14f64;
    let लंच = Π * Π + 1.54;
    assert!(((लंच - 1.54) - (Π * Π)).abs() < ε);
    assert_eq!(საჭმელად_გემრიელი_სადილი(), 0);
}

fn საჭმელად_გემრიელი_სადილი() -> int {

    // Lunch in several languages.

    let ランチ = 10;
    let 午餐 = 10;

    let ארוחת_צהריי = 10;
    let غداء = 10_usize;
    let լանչ = 10;
    let обед = 10;
    let абед = 10;
    let μεσημεριανό = 10;
    let hádegismatur = 10;
    let ручек = 10;

    let ăn_trưa = 10;
    let อาหารกลางวัน = 10;

    // Lunchy arithmetic, mm.

    assert_eq!(hádegismatur * ручек * обед, 1000);
    assert_eq!(10, ארוחת_צהריי);
    assert_eq!(ランチ + 午餐 + μεσημεριανό, 30);
    assert_eq!(ăn_trưa + อาหารกลางวัน, 20);
    return (абед + լանչ) >> غداء;
}
