#![feature(core_intrinsics)]

use std::io::Write;
use std::intrinsics;

fn checked_div_i128(lhs: i128, rhs: i128) -> Option<i128> {
    if rhs == 0 || (lhs == -170141183460469231731687303715884105728 && rhs == -1) {
        None
    } else {
        Some(unsafe { intrinsics::unchecked_div(lhs, rhs) })
    }
}

fn checked_div_u128(lhs: u128, rhs: u128) -> Option<u128> {
    match rhs {
        0 => None,
        rhs => Some(unsafe { intrinsics::unchecked_div(lhs, rhs) })
    }
}

fn main() {
    checked_div_i128(0i128, 2i128);
    checked_div_u128(0u128, 2u128);
    assert_eq!(1u128 + 2, 3);

    println!("{}", 0b100010000000000000000000000000000u128 >> 10);
    println!("{}", 0xFEDCBA987654321123456789ABCDEFu128 >> 64);
    println!("{} >> 64 == {}", 0xFEDCBA987654321123456789ABCDEFu128 as i128, 0xFEDCBA987654321123456789ABCDEFu128 as i128 >> 64);
    println!("{}", 353985398u128 * 932490u128);
}
