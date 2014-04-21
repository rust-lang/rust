// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules)]

use s = std::num::strconv;
use to_str = std::num::strconv::float_to_str_common;

macro_rules! t(($a:expr, $b:expr) => { { let (r, _) = $a; assert_eq!(r, $b.to_string()) } })

pub fn main() {
    // Basic usage
    t!(to_str(1.2345678e-5f64, 10u, true, s::SignNeg, s::DigMax(6), s::ExpDec, false),
             "1.234568e-5")

    // Hexadecimal output
    t!(to_str(7.281738281250e+01f64, 16u, true, s::SignAll, s::DigMax(6), s::ExpBin, false),
              "+1.2345p+6")
    t!(to_str(-1.777768135071e-02f64, 16u, true, s::SignAll, s::DigMax(6), s::ExpBin, false),
             "-1.2345p-6")

    // Some denormals
    t!(to_str(4.9406564584124654e-324f64, 10u, true, s::SignNeg, s::DigMax(6), s::ExpBin, false),
             "1p-1074")
    t!(to_str(2.2250738585072009e-308f64, 10u, true, s::SignNeg, s::DigMax(6), s::ExpBin, false),
             "1p-1022")
}
