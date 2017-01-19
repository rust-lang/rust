// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-stage0
// ignore-stage1

// ignore-emscripten

#![feature(i128_type, test)]

extern crate test;
use test::black_box as b;

fn main() {
    let x: u128 = 0xFFFF_FFFF_FFFF_FFFF__FFFF_FFFF_FFFF_FFFF;
    assert_eq!(0, !x);
    assert_eq!(0, !x);
    let y: u128 = 0xFFFF_FFFF_FFFF_FFFF__FFFF_FFFF_FFFF_FFFE;
    assert_eq!(!1, y);
    assert_eq!(x, y | 1);
    assert_eq!(0xFAFF_0000_FF8F_0000__FFFF_0000_FFFF_FFFE,
               y &
               0xFAFF_0000_FF8F_0000__FFFF_0000_FFFF_FFFF);
    let z: u128 = 0xABCD_EF;
    assert_eq!(z * z, 0x734C_C2F2_A521);
    assert_eq!(z * z * z * z, 0x33EE_0E2A_54E2_59DA_A0E7_8E41);
    assert_eq!(z + z + z + z, 0x2AF3_7BC);
    let k: u128 = 0x1234_5678_9ABC_DEFF_EDCB_A987_6543_210;
    assert_eq!(k + k, 0x2468_ACF1_3579_BDFF_DB97_530E_CA86_420);
    assert_eq!(0, k - k);
    assert_eq!(0x1234_5678_9ABC_DEFF_EDCB_A987_5A86_421, k - z);
    assert_eq!(0x1000_0000_0000_0000_0000_0000_0000_000,
               k - 0x234_5678_9ABC_DEFF_EDCB_A987_6543_210);
    assert_eq!(0x6EF5_DE4C_D3BC_2AAA_3BB4_CC5D_D6EE_8, k / 42);
    assert_eq!(0, k % 42);
    assert_eq!(15, z % 42);
    assert_eq!(0x169D_A8020_CEC18, k % 0x3ACB_FE49_FF24_AC);
    assert_eq!(0x91A2_B3C4_D5E6_F7, k >> 65);
    assert_eq!(0xFDB9_7530_ECA8_6420_0000_0000_0000_0000, k << 65);
    assert!(k > z);
    assert!(y > k);
    assert!(y < x);
    assert_eq!(x as u64, !0);
    assert_eq!(z as u64, 0xABCD_EF);
    assert_eq!(k as u64, 0xFEDC_BA98_7654_3210);
    assert_eq!(k as i128, 0x1234_5678_9ABC_DEFF_EDCB_A987_6543_210);
    assert_eq!((z as f64) as u128, z);
    assert_eq!((z as f32) as u128, z);
    assert_eq!((z as f64 * 16.0) as u128, z * 16);
    assert_eq!((z as f32 * 16.0) as u128, z * 16);
    let l :u128 = 432 << 100;
    assert_eq!((l as f32) as u128, l);
    assert_eq!((l as f64) as u128, l);
    // formatting
    let j: u128 = 1 << 67;
    assert_eq!("147573952589676412928", format!("{}", j));
    assert_eq!("80000000000000000", format!("{:x}", j));
    assert_eq!("20000000000000000000000", format!("{:o}", j));
    assert_eq!("10000000000000000000000000000000000000000000000000000000000000000000",
               format!("{:b}", j));
    assert_eq!("340282366920938463463374607431768211455",
        format!("{}", u128::max_value()));
    assert_eq!("147573952589676412928", format!("{:?}", j));
    // common traits
    assert_eq!(x, b(x.clone()));
    // overflow checks
    assert_eq!((z).checked_mul(z), Some(0x734C_C2F2_A521));
    assert_eq!((k).checked_mul(k), None);
    let l: u128 = b(u128::max_value() - 10);
    let o: u128 = b(17);
    assert_eq!(l.checked_add(b(11)), None);
    assert_eq!(l.checked_sub(l), Some(0));
    assert_eq!(o.checked_sub(b(18)), None);
    assert_eq!(b(1u128).checked_shl(b(127)), Some(1 << 127));
    assert_eq!(o.checked_shl(b(128)), None);
}
