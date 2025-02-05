//@ compile-flags: -C opt-level=3
//@ ignore-arm-unknown-linux-gnueabi very different IR compared to other targets
//@ ignore-arm-unknown-linux-gnueabihf
//@ ignore-arm-unknown-linux-musleabi
//@ ignore-arm-unknown-linux-musleabihf
//! Ensure that classifying `FpCategory` based on `usize` compiles to
//! a handful of instructions without any `switch` or branches.
//! This regressed between rustc 1.37 and 1.38 and was discovered
//! in issue #74615.

#![crate_type = "lib"]

use std::num::FpCategory;

fn conv(input: u32) -> FpCategory {
    match input {
        0b10000000 | 0b00000001 => FpCategory::Infinite,
        0b01000000 | 0b00000010 => FpCategory::Normal,
        0b00100000 | 0b00000100 => FpCategory::Subnormal,
        0b00010000 | 0b00001000 => FpCategory::Zero,
        0b100000000 | 0b1000000000 => FpCategory::Nan,
        _ => unsafe { std::hint::unreachable_unchecked() },
    }
}

#[no_mangle]
pub fn complex_test(input: u32) -> bool {
    // CHECK-LABEL: @complex_test(
    // CHECK:    [[TMP0:%.*]] = tail call
    // CHECK:    [[DOTOFF:%.*]] = add nsw i32 [[TMP0]], -3
    // CHECK:    [[SWITCH:%.*]] = icmp ult i32 [[DOTOFF]], 2
    // CHECK-NOT:   switch i32
    // CHECK-NOT:   br
    // CHECK-NOT:   label
    // CHECK:    ret i1 [[SWITCH]]
    //
    conv(input) == FpCategory::Zero
}

#[no_mangle]
pub fn simpler_test(input: u32) -> bool {
    // CHECK-LABEL: @simpler_test(
    // CHECK-SAME: i32 noundef [[INPUT:%.*]])
    // CHECK:    [[TMP0:%.*]] = add i32 [[INPUT]], -8
    // CHECK:    [[SWITCH_AND:%.*]] = and i32 [[TMP0]], -9
    // CHECK:    [[SWITCH_SELECTCMP:%.*]] = icmp eq i32 [[SWITCH_AND]], 0
    // CHECK-NOT:   switch i32
    // CHECK-NOT:   br
    // CHECK-NOT:   label
    // CHECK:    ret i1 [[SWITCH_SELECTCMP]]
    //
    match input {
        0b00010000 | 0b00001000 => true,
        _ => false,
    }
}
