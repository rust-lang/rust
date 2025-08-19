#![allow(clippy::legacy_numeric_constants, unused_imports)]

fn main() {
    let _ = 1u32.checked_add(1).unwrap_or(u32::max_value());
    //~^ manual_saturating_arithmetic
    let _ = 1u32.checked_add(1).unwrap_or(u32::MAX);
    //~^ manual_saturating_arithmetic
    let _ = 1u8.checked_add(1).unwrap_or(255);
    //~^ manual_saturating_arithmetic
    let _ = 1u128
        //~^ manual_saturating_arithmetic
        .checked_add(1)
        .unwrap_or(340_282_366_920_938_463_463_374_607_431_768_211_455);
    let _ = 1u32.checked_add(1).unwrap_or(1234); // ok
    let _ = 1u8.checked_add(1).unwrap_or(0); // ok
    let _ = 1u32.checked_mul(1).unwrap_or(u32::MAX);
    //~^ manual_saturating_arithmetic

    let _ = 1u32.checked_sub(1).unwrap_or(u32::min_value());
    //~^ manual_saturating_arithmetic
    let _ = 1u32.checked_sub(1).unwrap_or(u32::MIN);
    //~^ manual_saturating_arithmetic
    let _ = 1u8.checked_sub(1).unwrap_or(0);
    //~^ manual_saturating_arithmetic
    let _ = 1u32.checked_sub(1).unwrap_or(1234); // ok
    let _ = 1u8.checked_sub(1).unwrap_or(255); // ok

    let _ = 1i32.checked_add(1).unwrap_or(i32::max_value());
    //~^ manual_saturating_arithmetic
    let _ = 1i32.checked_add(1).unwrap_or(i32::MAX);
    //~^ manual_saturating_arithmetic
    let _ = 1i8.checked_add(1).unwrap_or(127);
    //~^ manual_saturating_arithmetic
    let _ = 1i128
        //~^ manual_saturating_arithmetic
        .checked_add(1)
        .unwrap_or(170_141_183_460_469_231_731_687_303_715_884_105_727);
    let _ = 1i32.checked_add(-1).unwrap_or(i32::min_value());
    //~^ manual_saturating_arithmetic
    let _ = 1i32.checked_add(-1).unwrap_or(i32::MIN);
    //~^ manual_saturating_arithmetic
    let _ = 1i8.checked_add(-1).unwrap_or(-128);
    //~^ manual_saturating_arithmetic
    let _ = 1i128
        //~^ manual_saturating_arithmetic
        .checked_add(-1)
        .unwrap_or(-170_141_183_460_469_231_731_687_303_715_884_105_728);
    let _ = 1i32.checked_add(1).unwrap_or(1234); // ok
    let _ = 1i8.checked_add(1).unwrap_or(-128); // ok
    let _ = 1i8.checked_add(-1).unwrap_or(127); // ok

    let _ = 1i32.checked_sub(1).unwrap_or(i32::min_value());
    //~^ manual_saturating_arithmetic
    let _ = 1i32.checked_sub(1).unwrap_or(i32::MIN);
    //~^ manual_saturating_arithmetic
    let _ = 1i8.checked_sub(1).unwrap_or(-128);
    //~^ manual_saturating_arithmetic
    let _ = 1i128
        //~^ manual_saturating_arithmetic
        .checked_sub(1)
        .unwrap_or(-170_141_183_460_469_231_731_687_303_715_884_105_728);
    let _ = 1i32.checked_sub(-1).unwrap_or(i32::max_value());
    //~^ manual_saturating_arithmetic
    let _ = 1i32.checked_sub(-1).unwrap_or(i32::MAX);
    //~^ manual_saturating_arithmetic
    let _ = 1i8.checked_sub(-1).unwrap_or(127);
    //~^ manual_saturating_arithmetic
    let _ = 1i128
        //~^ manual_saturating_arithmetic
        .checked_sub(-1)
        .unwrap_or(170_141_183_460_469_231_731_687_303_715_884_105_727);
    let _ = 1i32.checked_sub(1).unwrap_or(1234); // ok
    let _ = 1i8.checked_sub(1).unwrap_or(127); // ok
    let _ = 1i8.checked_sub(-1).unwrap_or(-128); // ok
}
