//@ test-mir-pass: MatchBranchSimplification

#![feature(repr128)]
#![feature(core_intrinsics)]
#![feature(custom_mir)]

use std::intrinsics::mir::*;

// EMIT_MIR matches_reduce_branches.foo.MatchBranchSimplification.diff
fn foo(bar: Option<()>) {
    // CHECK-LABEL: fn foo(
    // CHECK: = Eq(
    // CHECK: switchInt
    // CHECK-NOT: switchInt
    if matches!(bar, None) {
        ()
    }
}

// EMIT_MIR matches_reduce_branches.bar.MatchBranchSimplification.diff
fn bar(i: i32) -> (bool, bool, bool, bool) {
    // CHECK-LABEL: fn bar(
    // CHECK: = Ne(
    // CHECK: = Eq(
    // CHECK-NOT: switchInt
    let a;
    let b;
    let c;
    let d;

    match i {
        7 => {
            a = false;
            b = true;
            c = false;
            d = true;
            ()
        }
        _ => {
            a = true;
            b = false;
            c = false;
            d = true;
            ()
        }
    };

    (a, b, c, d)
}

// EMIT_MIR matches_reduce_branches.match_nested_if.MatchBranchSimplification.diff
fn match_nested_if() -> bool {
    // CHECK-LABEL: fn match_nested_if(
    // CHECK-NOT: switchInt
    let val = match () {
        () if if if if true { true } else { false } { true } else { false } {
            true
        } else {
            false
        } =>
        {
            true
        }
        _ => false,
    };
    val
}

#[repr(u8)]
enum EnumAu8 {
    A = 1,
    B = 2,
}

// EMIT_MIR matches_reduce_branches.match_u8_i16.MatchBranchSimplification.diff
fn match_u8_i16(i: EnumAu8) -> i16 {
    // CHECK-LABEL: fn match_u8_i16(
    // CHECK: switchInt
    match i {
        EnumAu8::A => 1,
        EnumAu8::B => 2,
    }
}

// EMIT_MIR matches_reduce_branches.match_u8_i16_2.MatchBranchSimplification.diff
// Check for different instruction lengths
#[custom_mir(dialect = "built")]
fn match_u8_i16_2(i: EnumAu8) -> i16 {
    // CHECK-LABEL: fn match_u8_i16_2(
    // CHECK: switchInt
    mir! {
        {
            let a = Discriminant(i);
            match a {
                1 => bb1,
                2 => bb2,
                _ => unreachable_bb,
            }
        }
        bb1 = {
            Goto(ret)
        }
        bb2 = {
            RET = 2;
            Goto(ret)
        }
        unreachable_bb = {
            Unreachable()
        }
        ret = {
            Return()
        }
    }
}

// EMIT_MIR matches_reduce_branches.match_u8_i16_failed.MatchBranchSimplification.diff
fn match_u8_i16_failed(i: EnumAu8) -> i16 {
    // CHECK-LABEL: fn match_u8_i16_failed(
    // CHECK: switchInt
    match i {
        EnumAu8::A => 1,
        EnumAu8::B => 3,
    }
}

// EMIT_MIR matches_reduce_branches.match_u8_i16_fallback.MatchBranchSimplification.diff
fn match_u8_i16_fallback(i: u8) -> i16 {
    // CHECK-LABEL: fn match_u8_i16_fallback(
    // CHECK: switchInt
    match i {
        1 => 1,
        2 => 2,
        _ => 3,
    }
}

#[repr(u8)]
enum EnumBu8 {
    A = 1,
    B = 2,
    C = 5,
}

// EMIT_MIR matches_reduce_branches.match_u8_u16.MatchBranchSimplification.diff
fn match_u8_u16(i: EnumBu8) -> u16 {
    // CHECK-LABEL: fn match_u8_u16(
    // CHECK: switchInt
    match i {
        EnumBu8::A => 1,
        EnumBu8::B => 2,
        EnumBu8::C => 5,
    }
}

// EMIT_MIR matches_reduce_branches.match_u8_u16_2.MatchBranchSimplification.diff
// Check for different instruction lengths
#[custom_mir(dialect = "built")]
fn match_u8_u16_2(i: EnumBu8) -> i16 {
    // CHECK-LABEL: fn match_u8_u16_2(
    // CHECK: switchInt
    mir! {
        {
            let a = Discriminant(i);
            match a {
                1 => bb1,
                2 => bb2,
                5 => bb5,
                _ => unreachable_bb,
            }
        }
        bb1 = {
            RET = 1;
            Goto(ret)
        }
        bb2 = {
            RET = 2;
            Goto(ret)
        }
        bb5 = {
            RET = 5;
            RET = 5;
            Goto(ret)
        }
        unreachable_bb = {
            Unreachable()
        }
        ret = {
            Return()
        }
    }
}

#[repr(i8)]
enum EnumAi8 {
    A = -1,
    B = 2,
    C = -3,
}

// EMIT_MIR matches_reduce_branches.match_i8_i16.MatchBranchSimplification.diff
fn match_i8_i16(i: EnumAi8) -> i16 {
    // CHECK-LABEL: fn match_i8_i16(
    // CHECK: switchInt
    match i {
        EnumAi8::A => -1,
        EnumAi8::B => 2,
        EnumAi8::C => -3,
    }
}

// EMIT_MIR matches_reduce_branches.match_i8_i16_failed.MatchBranchSimplification.diff
fn match_i8_i16_failed(i: EnumAi8) -> i16 {
    // CHECK-LABEL: fn match_i8_i16_failed(
    // CHECK: switchInt
    match i {
        EnumAi8::A => -1,
        EnumAi8::B => 2,
        EnumAi8::C => 3,
    }
}

#[repr(i16)]
enum EnumAi16 {
    A = -1,
    B = 2,
    C = -3,
}

// EMIT_MIR matches_reduce_branches.match_i16_i8.MatchBranchSimplification.diff
fn match_i16_i8(i: EnumAi16) -> i8 {
    // CHECK-LABEL: fn match_i16_i8(
    // CHECK: switchInt
    match i {
        EnumAi16::A => -1,
        EnumAi16::B => 2,
        EnumAi16::C => -3,
    }
}

#[repr(i128)]
enum EnumAi128 {
    A = 1,
    B = 2,
    C = 3,
    D = -1,
}

// EMIT_MIR matches_reduce_branches.match_i128_u128.MatchBranchSimplification.diff
fn match_i128_u128(i: EnumAi128) -> u128 {
    // CHECK-LABEL: fn match_i128_u128(
    // CHECK: switchInt
    match i {
        EnumAi128::A => 1,
        EnumAi128::B => 2,
        EnumAi128::C => 3,
        EnumAi128::D => u128::MAX,
    }
}

fn main() {
    let _ = foo(None);
    let _ = foo(Some(()));
    let _ = bar(0);
    let _ = match_nested_if();
    let _ = match_u8_i16(EnumAu8::A);
    let _ = match_u8_i16_2(EnumAu8::A);
    let _ = match_u8_i16_failed(EnumAu8::A);
    let _ = match_u8_i16_fallback(1);
    let _ = match_u8_u16(EnumBu8::A);
    let _ = match_u8_u16_2(EnumBu8::A);
    let _ = match_i8_i16(EnumAi8::A);
    let _ = match_i8_i16_failed(EnumAi8::A);
    let _ = match_i8_i16(EnumAi8::A);
    let _ = match_i16_i8(EnumAi16::A);
    let _ = match_i128_u128(EnumAi128::A);
}
