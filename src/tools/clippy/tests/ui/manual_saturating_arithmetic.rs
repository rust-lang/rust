// run-rustfix

#![allow(unused_imports)]

use std::{i128, i32, u128, u32};

fn main() {
    let _ = 1u32.checked_add(1).unwrap_or(u32::max_value());
    let _ = 1u32.checked_add(1).unwrap_or(u32::MAX);
    let _ = 1u8.checked_add(1).unwrap_or(255);
    let _ = 1u128
        .checked_add(1)
        .unwrap_or(340_282_366_920_938_463_463_374_607_431_768_211_455);
    let _ = 1u32.checked_add(1).unwrap_or(1234); // ok
    let _ = 1u8.checked_add(1).unwrap_or(0); // ok
    let _ = 1u32.checked_mul(1).unwrap_or(u32::MAX);

    let _ = 1u32.checked_sub(1).unwrap_or(u32::min_value());
    let _ = 1u32.checked_sub(1).unwrap_or(u32::MIN);
    let _ = 1u8.checked_sub(1).unwrap_or(0);
    let _ = 1u32.checked_sub(1).unwrap_or(1234); // ok
    let _ = 1u8.checked_sub(1).unwrap_or(255); // ok

    let _ = 1i32.checked_add(1).unwrap_or(i32::max_value());
    let _ = 1i32.checked_add(1).unwrap_or(i32::MAX);
    let _ = 1i8.checked_add(1).unwrap_or(127);
    let _ = 1i128
        .checked_add(1)
        .unwrap_or(170_141_183_460_469_231_731_687_303_715_884_105_727);
    let _ = 1i32.checked_add(-1).unwrap_or(i32::min_value());
    let _ = 1i32.checked_add(-1).unwrap_or(i32::MIN);
    let _ = 1i8.checked_add(-1).unwrap_or(-128);
    let _ = 1i128
        .checked_add(-1)
        .unwrap_or(-170_141_183_460_469_231_731_687_303_715_884_105_728);
    let _ = 1i32.checked_add(1).unwrap_or(1234); // ok
    let _ = 1i8.checked_add(1).unwrap_or(-128); // ok
    let _ = 1i8.checked_add(-1).unwrap_or(127); // ok

    let _ = 1i32.checked_sub(1).unwrap_or(i32::min_value());
    let _ = 1i32.checked_sub(1).unwrap_or(i32::MIN);
    let _ = 1i8.checked_sub(1).unwrap_or(-128);
    let _ = 1i128
        .checked_sub(1)
        .unwrap_or(-170_141_183_460_469_231_731_687_303_715_884_105_728);
    let _ = 1i32.checked_sub(-1).unwrap_or(i32::max_value());
    let _ = 1i32.checked_sub(-1).unwrap_or(i32::MAX);
    let _ = 1i8.checked_sub(-1).unwrap_or(127);
    let _ = 1i128
        .checked_sub(-1)
        .unwrap_or(170_141_183_460_469_231_731_687_303_715_884_105_727);
    let _ = 1i32.checked_sub(1).unwrap_or(1234); // ok
    let _ = 1i8.checked_sub(1).unwrap_or(127); // ok
    let _ = 1i8.checked_sub(-1).unwrap_or(-128); // ok
}
