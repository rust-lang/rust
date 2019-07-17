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
}
