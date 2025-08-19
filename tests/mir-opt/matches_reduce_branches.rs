//@ test-mir-pass: MatchBranchSimplification

#![feature(core_intrinsics)]
#![feature(custom_mir)]
#![allow(non_camel_case_types)]

use std::intrinsics::mir::*;

// EMIT_MIR matches_reduce_branches.foo.MatchBranchSimplification.diff
fn foo(bar: Option<()>) {
    // CHECK-LABEL: fn foo(
    // CHECK: = Eq(
    // CHECK: switchInt
    // CHECK-NOT: switchInt
    // CHECK: return
    if matches!(bar, None) {
        ()
    }
}

// EMIT_MIR matches_reduce_branches.my_is_some.MatchBranchSimplification.diff
// Test for #131219.
fn my_is_some(bar: Option<()>) -> bool {
    // CHECK-LABEL: fn my_is_some(
    // CHECK: = Ne
    // CHECK: return
    match bar {
        Some(_) => true,
        None => false,
    }
}

// EMIT_MIR matches_reduce_branches.bar.MatchBranchSimplification.diff
fn bar(i: i32) -> (bool, bool, bool, bool) {
    // CHECK-LABEL: fn bar(
    // CHECK: = Ne(
    // CHECK: = Eq(
    // CHECK-NOT: switchInt
    // CHECK: return
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
    // CHECK: return
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

// # Fold switchInt into IntToInt.
// To simplify writing and checking these test cases, I use the first character of
// each case to distinguish the sign of the number:
// 'u' for unsigned, '_' for negative, and 'o' for non-negative.
// Followed by a decimal number, and add the corresponding radix representation.
// For example, o127_0x7f represents 127i8, and _1_0xff represents -1i8.

// ## Cast but without numeric conversion.

#[repr(u8)]
enum EnumAu8 {
    u0_0x00 = 0,
    u127_0x7f = 127,
    u128_0x80 = 128,
    u255_0xff = 255,
}

#[repr(i8)]
enum EnumAi8 {
    _128_0x80 = -128,
    _1_0xff = -1,
    o0_0x00 = 0,
    o1_0x01 = 1,
    o127_0x7f = 127,
}

// EMIT_MIR matches_reduce_branches.match_u8_i8.MatchBranchSimplification.diff
fn match_u8_i8(i: EnumAu8) -> i8 {
    // CHECK-LABEL: fn match_u8_i8(
    // CHECK-NOT: switchInt
    // CHECK: return
    match i {
        EnumAu8::u0_0x00 => 0,
        EnumAu8::u127_0x7f => 127,
        EnumAu8::u128_0x80 => -128,
        EnumAu8::u255_0xff => -1,
    }
}

// EMIT_MIR matches_reduce_branches.match_u8_i8_failed.MatchBranchSimplification.diff
fn match_u8_i8_failed(i: EnumAu8) -> i8 {
    // CHECK-LABEL: fn match_u8_i8_failed(
    // CHECK: switchInt
    // CHECK: return
    match i {
        EnumAu8::u0_0x00 => 0,
        EnumAu8::u127_0x7f => 127,
        EnumAu8::u128_0x80 => -128,
        EnumAu8::u255_0xff => 1, // failed
    }
}

// EMIT_MIR matches_reduce_branches.match_u8_i8_2.MatchBranchSimplification.diff
fn match_u8_i8_2(i: EnumAu8) -> (i8, i8) {
    // CHECK-LABEL: fn match_u8_i8_2(
    // CHECK-NOT: switchInt
    // CHECK: IntToInt
    // CHECK: IntToInt
    // CHECK: return
    let a: i8;
    let b: i8;
    match i {
        EnumAu8::u0_0x00 => {
            a = 0;
            b = 0;
            ()
        }
        EnumAu8::u127_0x7f => {
            a = 127;
            b = 127;
            ()
        }
        EnumAu8::u128_0x80 => {
            a = -128;
            b = -128;
            ()
        }
        EnumAu8::u255_0xff => {
            a = -1;
            b = -1;
            ()
        }
    };
    (a, b)
}

// EMIT_MIR matches_reduce_branches.match_u8_i8_2_failed.MatchBranchSimplification.diff
fn match_u8_i8_2_failed(i: EnumAu8) -> (i8, i8) {
    // CHECK-LABEL: fn match_u8_i8_2_failed(
    // CHECK: switchInt
    // CHECK: return
    let a: i8;
    let b: i8;
    match i {
        EnumAu8::u0_0x00 => {
            a = 0;
            b = 0;
            ()
        }
        EnumAu8::u127_0x7f => {
            a = 127;
            b = 127;
            ()
        }
        EnumAu8::u128_0x80 => {
            a = -128;
            b = -128;
            ()
        }
        EnumAu8::u255_0xff => {
            a = -1;
            b = 1; // failed
            ()
        }
    };
    (a, b)
}

// EMIT_MIR matches_reduce_branches.match_u8_i8_fallback.MatchBranchSimplification.diff
fn match_u8_i8_fallback(i: EnumAu8) -> i8 {
    // CHECK-LABEL: fn match_u8_i8_fallback(
    // CHECK: switchInt
    // CHECK: return
    match i {
        EnumAu8::u0_0x00 => 0,
        EnumAu8::u127_0x7f => 127,
        EnumAu8::u128_0x80 => -128,
        _ => -1,
    }
}

// EMIT_MIR matches_reduce_branches.match_u8_i8_failed_len_1.MatchBranchSimplification.diff
// Check for different instruction lengths
#[custom_mir(dialect = "built")]
fn match_u8_i8_failed_len_1(i: EnumAu8) -> i8 {
    // CHECK-LABEL: fn match_u8_i8_failed_len_1(
    // CHECK: switchInt
    // CHECK: return
    mir! {
        {
            let a = Discriminant(i);
            match a {
                0 => bb1,
                127 => bb2,
                128 => bb3,
                255 => bb4,
                _ => unreachable_bb,
            }
        }
        bb1 = {
            RET = 0;
            Goto(ret)
        }
        bb2 = {
            RET = 127;
            RET = 127;
            Goto(ret)
        }
        bb3 = {
            RET = -128;
            Goto(ret)
        }
        bb4 = {
            RET = -1;
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

// EMIT_MIR matches_reduce_branches.match_u8_i8_failed_len_2.MatchBranchSimplification.diff
// Check for different instruction lengths
#[custom_mir(dialect = "built")]
fn match_u8_i8_failed_len_2(i: EnumAu8) -> i8 {
    // CHECK-LABEL: fn match_u8_i8_failed_len_2(
    // CHECK: switchInt
    // CHECK: return
    mir! {
        {
            let a = Discriminant(i);
            match a {
                0 => bb1,
                127 => bb2,
                128 => bb3,
                255 => bb4,
                _ => unreachable_bb,
            }
        }
        bb1 = {
            RET = 0;
            Goto(ret)
        }
        bb2 = {
            RET = 127;
            Goto(ret)
        }
        bb3 = {
            RET = -128;
            Goto(ret)
        }
        bb4 = {
            RET = -1;
            RET = -1;
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

// ## Cast with sext.

// EMIT_MIR matches_reduce_branches.match_sext_i8_i16.MatchBranchSimplification.diff
fn match_sext_i8_i16(i: EnumAi8) -> i16 {
    // CHECK-LABEL: fn match_sext_i8_i16(
    // CHECK-NOT: switchInt
    // CHECK: return
    match i {
        EnumAi8::_128_0x80 => -128,
        EnumAi8::_1_0xff => -1,
        EnumAi8::o0_0x00 => 0,
        EnumAi8::o1_0x01 => 1,
        EnumAi8::o127_0x7f => 127,
    }
}

// EMIT_MIR matches_reduce_branches.match_sext_i8_i16_failed.MatchBranchSimplification.diff
// Converting `-1i8` to `255i16` is zext.
fn match_sext_i8_i16_failed(i: EnumAi8) -> i16 {
    // CHECK-LABEL: fn match_sext_i8_i16_failed(
    // CHECK: switchInt
    // CHECK: return
    match i {
        EnumAi8::_128_0x80 => -128,
        EnumAi8::_1_0xff => 255, // failed
        EnumAi8::o0_0x00 => 0,
        EnumAi8::o1_0x01 => 1,
        EnumAi8::o127_0x7f => 127,
    }
}

// EMIT_MIR matches_reduce_branches.match_sext_i8_u16.MatchBranchSimplification.diff
fn match_sext_i8_u16(i: EnumAi8) -> u16 {
    // CHECK-LABEL: fn match_sext_i8_u16(
    // CHECK-NOT: switchInt
    // CHECK: return
    match i {
        EnumAi8::_128_0x80 => 0xff80,
        EnumAi8::_1_0xff => 0xffff,
        EnumAi8::o0_0x00 => 0,
        EnumAi8::o1_0x01 => 1,
        EnumAi8::o127_0x7f => 127,
    }
}

// EMIT_MIR matches_reduce_branches.match_sext_i8_u16_failed.MatchBranchSimplification.diff
// Converting `-1i8` to `255u16` is zext.
fn match_sext_i8_u16_failed(i: EnumAi8) -> u16 {
    // CHECK-LABEL: fn match_sext_i8_u16_failed(
    // CHECK: switchInt
    // CHECK: return
    match i {
        EnumAi8::_128_0x80 => 0xff80,
        EnumAi8::_1_0xff => 0x00ff, // failed
        EnumAi8::o0_0x00 => 0,
        EnumAi8::o1_0x01 => 1,
        EnumAi8::o127_0x7f => 127,
    }
}

// ## Cast with zext.

// EMIT_MIR matches_reduce_branches.match_zext_u8_u16.MatchBranchSimplification.diff
fn match_zext_u8_u16(i: EnumAu8) -> u16 {
    // CHECK-LABEL: fn match_zext_u8_u16(
    // CHECK-NOT: switchInt
    // CHECK: return
    match i {
        EnumAu8::u0_0x00 => 0,
        EnumAu8::u127_0x7f => 0x7f,
        EnumAu8::u128_0x80 => 128,
        EnumAu8::u255_0xff => 255,
    }
}

// EMIT_MIR matches_reduce_branches.match_zext_u8_u16_failed.MatchBranchSimplification.diff
fn match_zext_u8_u16_failed(i: EnumAu8) -> u16 {
    // CHECK-LABEL: fn match_zext_u8_u16_failed(
    // CHECK: switchInt
    // CHECK: return
    match i {
        EnumAu8::u0_0x00 => 0,
        EnumAu8::u127_0x7f => 0xff7f, // failed
        EnumAu8::u128_0x80 => 128,
        EnumAu8::u255_0xff => 255,
    }
}

// EMIT_MIR matches_reduce_branches.match_zext_u8_i16.MatchBranchSimplification.diff
fn match_zext_u8_i16(i: EnumAu8) -> i16 {
    // CHECK-LABEL: fn match_zext_u8_i16(
    // CHECK-NOT: switchInt
    // CHECK: return
    match i {
        EnumAu8::u0_0x00 => 0,
        EnumAu8::u127_0x7f => 127,
        EnumAu8::u128_0x80 => 128,
        EnumAu8::u255_0xff => 255,
    }
}

// EMIT_MIR matches_reduce_branches.match_zext_u8_i16_failed.MatchBranchSimplification.diff
fn match_zext_u8_i16_failed(i: EnumAu8) -> i16 {
    // CHECK-LABEL: fn match_zext_u8_i16_failed(
    // CHECK: switchInt
    // CHECK: return
    match i {
        EnumAu8::u0_0x00 => 0,
        EnumAu8::u127_0x7f => -127, // failed
        EnumAu8::u128_0x80 => 128,
        EnumAu8::u255_0xff => 255,
    }
}

// ## Cast with trunc.

#[repr(u16)]
enum EnumAu16 {
    // 0x00nn
    u0_0x0000 = 0,
    u127_0x007f = 127,
    u128_0x0080 = 128,
    u255_0x00ff = 255,
    // 0xffnn
    u65280_0xff00 = 65280,
    u65407_0xff7f = 65407,
    u65408_0xff80 = 65408,
    u65535_0xffff = 65535,
}

#[repr(i16)]
enum EnumAi16 {
    // 0x00nn
    o128_0x0080 = 128,
    o255_0x00ff = 255,
    o0_0x0000 = 0,
    o1_0x0001 = 1,
    o127_0x007f = 127,
    // 0xffnn
    _128_0xff80 = -128,
    _1_0xffff = -1,
    o0_0xff00 = -256,
    o1_0xff01 = -255,
    o127_0xff7f = -129,
}

// EMIT_MIR matches_reduce_branches.match_trunc_i16_i8.MatchBranchSimplification.diff
fn match_trunc_i16_i8(i: EnumAi16) -> i8 {
    // CHECK-LABEL: fn match_trunc_i16_i8(
    // CHECK-NOT: switchInt
    // CHECK: return
    match i {
        // 0x00nn
        EnumAi16::o128_0x0080 => -128,
        EnumAi16::o255_0x00ff => -1,
        EnumAi16::o0_0x0000 => 0,
        EnumAi16::o1_0x0001 => 1,
        EnumAi16::o127_0x007f => 127,
        // 0xffnn
        EnumAi16::_128_0xff80 => -128,
        EnumAi16::_1_0xffff => -1,
        EnumAi16::o0_0xff00 => 0,
        EnumAi16::o1_0xff01 => 1,
        EnumAi16::o127_0xff7f => 127,
    }
}

// EMIT_MIR matches_reduce_branches.match_trunc_i16_i8_failed.MatchBranchSimplification.diff
fn match_trunc_i16_i8_failed(i: EnumAi16) -> i8 {
    // CHECK-LABEL: fn match_trunc_i16_i8_failed(
    // CHECK: switchInt
    // CHECK: return
    match i {
        // 0x00nn
        EnumAi16::o128_0x0080 => -128,
        EnumAi16::o255_0x00ff => -1,
        EnumAi16::o0_0x0000 => 0,
        EnumAi16::o1_0x0001 => 1,
        EnumAi16::o127_0x007f => 127,
        // 0xffnn
        EnumAi16::_128_0xff80 => -128,
        EnumAi16::_1_0xffff => -1,
        EnumAi16::o0_0xff00 => 0,
        EnumAi16::o1_0xff01 => 1,
        EnumAi16::o127_0xff7f => -127, // failed
    }
}

// EMIT_MIR matches_reduce_branches.match_trunc_i16_u8.MatchBranchSimplification.diff
fn match_trunc_i16_u8(i: EnumAi16) -> u8 {
    // CHECK-LABEL: fn match_trunc_i16_u8(
    // CHECK-NOT: switchInt
    // CHECK: return
    match i {
        // 0x00nn
        EnumAi16::o128_0x0080 => 128,
        EnumAi16::o255_0x00ff => 255,
        EnumAi16::o0_0x0000 => 0,
        EnumAi16::o1_0x0001 => 1,
        EnumAi16::o127_0x007f => 127,
        // 0xffnn
        EnumAi16::_128_0xff80 => 128,
        EnumAi16::_1_0xffff => 255,
        EnumAi16::o0_0xff00 => 0,
        EnumAi16::o1_0xff01 => 1,
        EnumAi16::o127_0xff7f => 127,
    }
}

// EMIT_MIR matches_reduce_branches.match_trunc_i16_u8_failed.MatchBranchSimplification.diff
fn match_trunc_i16_u8_failed(i: EnumAi16) -> u8 {
    // CHECK-LABEL: fn match_trunc_i16_u8_failed(
    // CHECK: switchInt
    // CHECK: return
    match i {
        // 0x00nn
        EnumAi16::o128_0x0080 => 128,
        EnumAi16::o255_0x00ff => 255,
        EnumAi16::o0_0x0000 => 0,
        EnumAi16::o1_0x0001 => 1,
        EnumAi16::o127_0x007f => 127,
        // 0xffnn
        EnumAi16::_128_0xff80 => 128,
        EnumAi16::_1_0xffff => 255,
        EnumAi16::o0_0xff00 => 0,
        EnumAi16::o1_0xff01 => 1,
        EnumAi16::o127_0xff7f => -127i8 as u8, // failed
    }
}

// EMIT_MIR matches_reduce_branches.match_trunc_u16_u8.MatchBranchSimplification.diff
fn match_trunc_u16_u8(i: EnumAu16) -> u8 {
    // CHECK-LABEL: fn match_trunc_u16_u8(
    // CHECK-NOT: switchInt
    // CHECK: return
    match i {
        // 0x00nn
        EnumAu16::u0_0x0000 => 0,
        EnumAu16::u127_0x007f => 127,
        EnumAu16::u128_0x0080 => 128,
        EnumAu16::u255_0x00ff => 255,
        // 0xffnn
        EnumAu16::u65280_0xff00 => 0,
        EnumAu16::u65407_0xff7f => 127,
        EnumAu16::u65408_0xff80 => 128,
        EnumAu16::u65535_0xffff => 255,
    }
}

// EMIT_MIR matches_reduce_branches.match_trunc_u16_u8_failed.MatchBranchSimplification.diff
fn match_trunc_u16_u8_failed(i: EnumAu16) -> u8 {
    // CHECK-LABEL: fn match_trunc_u16_u8_failed(
    // CHECK: switchInt
    // CHECK: return
    match i {
        // 0x00nn
        EnumAu16::u0_0x0000 => 0,
        EnumAu16::u127_0x007f => 127,
        EnumAu16::u128_0x0080 => 128,
        EnumAu16::u255_0x00ff => 255,
        // 0xffnn
        EnumAu16::u65280_0xff00 => 0,
        EnumAu16::u65407_0xff7f => 127,
        EnumAu16::u65408_0xff80 => 128,
        EnumAu16::u65535_0xffff => 127, // failed
    }
}

// EMIT_MIR matches_reduce_branches.match_trunc_u16_i8.MatchBranchSimplification.diff
fn match_trunc_u16_i8(i: EnumAu16) -> i8 {
    // CHECK-LABEL: fn match_trunc_u16_i8(
    // CHECK-NOT: switchInt
    // CHECK: return
    match i {
        // 0x00nn
        EnumAu16::u0_0x0000 => 0,
        EnumAu16::u127_0x007f => 127,
        EnumAu16::u128_0x0080 => -128,
        EnumAu16::u255_0x00ff => -1,
        // 0xffnn
        EnumAu16::u65280_0xff00 => 0,
        EnumAu16::u65407_0xff7f => 127,
        EnumAu16::u65408_0xff80 => -128,
        EnumAu16::u65535_0xffff => -1,
    }
}

// EMIT_MIR matches_reduce_branches.match_trunc_u16_i8_failed.MatchBranchSimplification.diff
fn match_trunc_u16_i8_failed(i: EnumAu16) -> i8 {
    // CHECK-LABEL: fn match_trunc_u16_i8_failed(
    // CHECK: switchInt
    // CHECK: return
    match i {
        // 0x00nn
        EnumAu16::u0_0x0000 => 0,
        EnumAu16::u127_0x007f => 127,
        EnumAu16::u128_0x0080 => -128,
        EnumAu16::u255_0x00ff => -1,
        // 0xffnn
        EnumAu16::u65280_0xff00 => 0,
        EnumAu16::u65407_0xff7f => 127,
        EnumAu16::u65408_0xff80 => -128,
        EnumAu16::u65535_0xffff => 1,
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
    // CHECK-NOT: switchInt
    // CHECK: return
    match i {
        EnumAi128::A => 1,
        EnumAi128::B => 2,
        EnumAi128::C => 3,
        EnumAi128::D => u128::MAX,
    }
}

// EMIT_MIR matches_reduce_branches.match_non_int_failed.MatchBranchSimplification.diff
#[custom_mir(dialect = "runtime")]
fn match_non_int_failed(i: char) -> u8 {
    // CHECK-LABEL: fn match_non_int_failed(
    // CHECK: switchInt
    // CHECK: return
    mir! {
        {
            match i {
                'a' => bb1,
                'b' => bb2,
                _ => unreachable_bb,
            }
        }
        bb1 = {
            RET = 97;
            Goto(ret)
        }
        bb2 = {
            RET = 98;
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

fn main() {
    let _ = foo(None);
    let _ = foo(Some(()));
    let _ = bar(0);
    let _ = match_nested_if();

    let _: i8 = match_u8_i8(EnumAu8::u0_0x00);
    let _: i8 = match_u8_i8_failed(EnumAu8::u0_0x00);
    let _: (i8, i8) = match_u8_i8_2(EnumAu8::u0_0x00);
    let _: (i8, i8) = match_u8_i8_2_failed(EnumAu8::u0_0x00);
    let _: i8 = match_u8_i8_fallback(EnumAu8::u0_0x00);
    let _: i8 = match_u8_i8_failed_len_1(EnumAu8::u0_0x00);
    let _: i8 = match_u8_i8_failed_len_2(EnumAu8::u0_0x00);

    let _: i16 = match_sext_i8_i16(EnumAi8::o0_0x00);
    let _: i16 = match_sext_i8_i16_failed(EnumAi8::o0_0x00);
    let _: u16 = match_sext_i8_u16(EnumAi8::o0_0x00);
    let _: u16 = match_sext_i8_u16_failed(EnumAi8::o0_0x00);

    let _: u16 = match_zext_u8_u16(EnumAu8::u0_0x00);
    let _: u16 = match_zext_u8_u16_failed(EnumAu8::u0_0x00);
    let _: i16 = match_zext_u8_i16(EnumAu8::u0_0x00);
    let _: i16 = match_zext_u8_i16_failed(EnumAu8::u0_0x00);

    let _: i8 = match_trunc_i16_i8(EnumAi16::o0_0x0000);
    let _: i8 = match_trunc_i16_i8_failed(EnumAi16::o0_0x0000);
    let _: u8 = match_trunc_i16_u8(EnumAi16::o0_0x0000);
    let _: u8 = match_trunc_i16_u8_failed(EnumAi16::o0_0x0000);

    let _: i8 = match_trunc_u16_i8(EnumAu16::u0_0x0000);
    let _: i8 = match_trunc_u16_i8_failed(EnumAu16::u0_0x0000);
    let _: u8 = match_trunc_u16_u8(EnumAu16::u0_0x0000);
    let _: u8 = match_trunc_u16_u8_failed(EnumAu16::u0_0x0000);

    let _ = match_i128_u128(EnumAi128::A);

    let _ = my_is_some(None);
    let _ = match_non_int_failed('a');
}
