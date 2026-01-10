//@ test-mir-pass: SimplifyComparisonIntegral
// GVN simplifies FileCheck.
//@ compile-flags: -Zmir-enable-passes=+GVN

#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

// EMIT_MIR if_condition_int.opt_u32.SimplifyComparisonIntegral.diff
fn opt_u32(x: u32) -> u32 {
    // CHECK-LABEL: fn opt_u32(
    // CHECK: switchInt(copy _1) -> [42: [[BB1:bb.*]], otherwise: [[BB2:bb.*]]];
    // CHECK: [[BB1]]:
    // CHECK: _0 = const 0_u32;
    // CHECK: [[BB2]]:
    // CHECK: _0 = const 1_u32;
    if x == 42 { 0 } else { 1 }
}

// EMIT_MIR if_condition_int.dont_opt_bool.SimplifyComparisonIntegral.diff
// don't opt: it is already optimal to switch on the bool
fn dont_opt_bool(x: bool) -> u32 {
    // CHECK-LABEL: fn dont_opt_bool(
    // CHECK: switchInt(copy _1) -> [0: [[BB2:bb.*]], otherwise: [[BB1:bb.*]]];
    // CHECK: [[BB1]]:
    // CHECK: _0 = const 0_u32;
    // CHECK: [[BB2]]:
    // CHECK: _0 = const 1_u32;
    if x { 0 } else { 1 }
}

// EMIT_MIR if_condition_int.opt_char.SimplifyComparisonIntegral.diff
fn opt_char(x: char) -> u32 {
    // CHECK-LABEL: fn opt_char(
    // CHECK: switchInt(copy _1) -> [120: [[BB1:bb.*]], otherwise: [[BB2:bb.*]]];
    // CHECK: [[BB1]]:
    // CHECK: _0 = const 0_u32;
    // CHECK: [[BB2]]:
    // CHECK: _0 = const 1_u32;
    if x == 'x' { 0 } else { 1 }
}

// EMIT_MIR if_condition_int.opt_i8.SimplifyComparisonIntegral.diff
fn opt_i8(x: i8) -> u32 {
    // CHECK-LABEL: fn opt_i8(
    // CHECK: switchInt(copy _1) -> [42: [[BB1:bb.*]], otherwise: [[BB2:bb.*]]];
    // CHECK: [[BB1]]:
    // CHECK: _0 = const 0_u32;
    // CHECK: [[BB2]]:
    // CHECK: _0 = const 1_u32;
    if x == 42 { 0 } else { 1 }
}

// EMIT_MIR if_condition_int.opt_negative.SimplifyComparisonIntegral.diff
fn opt_negative(x: i32) -> u32 {
    // CHECK-LABEL: fn opt_negative(
    // CHECK: switchInt(copy _1) -> [4294967254: [[BB1:bb.*]], otherwise: [[BB2:bb.*]]];
    // CHECK: [[BB1]]:
    // CHECK: _0 = const 0_u32;
    // CHECK: [[BB2]]:
    // CHECK: _0 = const 1_u32;
    if x == -42 { 0 } else { 1 }
}

// EMIT_MIR if_condition_int.opt_multiple_ifs.SimplifyComparisonIntegral.diff
fn opt_multiple_ifs(x: u32) -> u32 {
    // CHECK-LABEL: fn opt_multiple_ifs(
    // CHECK: switchInt(copy _1) -> [42: [[BB1:bb.*]], otherwise: [[BB2:bb.*]]];
    // CHECK: [[BB1]]:
    // CHECK: _0 = const 0_u32;
    // CHECK: [[BB2]]:
    // CHECK: switchInt(copy _1) -> [21: [[BB4:bb.*]], otherwise: [[BB3:bb.*]]];
    // CHECK: [[BB3]]:
    // CHECK: _0 = const 1_u32;
    // CHECK: [[BB4]]:
    // CHECK: _0 = const 2_u32;
    if x == 42 {
        0
    } else if x != 21 {
        1
    } else {
        2
    }
}

// EMIT_MIR if_condition_int.dont_remove_comparison.SimplifyComparisonIntegral.diff
// test that we optimize, but do not remove the b statement, as that is used later on
fn dont_remove_comparison(a: i8) -> i32 {
    // CHECK-LABEL: fn dont_remove_comparison(
    // CHECK: [[b:_.*]] = Eq(copy _1, const 17_i8);
    // CHECK: switchInt(copy _1) -> [17: [[BB1:bb.*]], otherwise: [[BB2:bb.*]]];
    // CHECK: [[BB1]]:
    // CHECK: [[cast_1:_.*]] = copy [[b]] as i32 (IntToInt);
    // CHECK: _0 = Add(const 100_i32, move [[cast_1]]);
    // CHECK: [[BB2]]:
    // CHECK: [[cast_2:_.*]] = copy [[b]] as i32 (IntToInt);
    // CHECK: _0 = Add(const 10_i32, move [[cast_2]]);
    let b = a == 17;
    match b {
        false => 10 + b as i32,
        true => 100 + b as i32,
    }
}

// EMIT_MIR if_condition_int.dont_opt_floats.SimplifyComparisonIntegral.diff
// test that we do not optimize on floats
fn dont_opt_floats(a: f32) -> i32 {
    // CHECK-LABEL: fn dont_opt_floats(
    // CHECK: [[cmp:_.*]] = Eq(copy _1, const -42f32);
    // CHECK: switchInt(move [[cmp]]) -> [0: [[BB2:bb.*]], otherwise: [[BB1:bb.*]]];
    // CHECK: [[BB1]]:
    // CHECK: _0 = const 0_i32;
    // CHECK: [[BB2]]:
    // CHECK: _0 = const 1_i32;
    if a == -42.0 { 0 } else { 1 }
}

// EMIT_MIR if_condition_int.on_non_ssa_switch.SimplifyComparisonIntegral.diff
#[custom_mir(dialect = "runtime")]
pub fn on_non_ssa_switch(mut v: u64) -> i32 {
    // CHECK-LABEL: fn on_non_ssa_switch(
    // CHECK: [[cmp:_.*]] = Eq(copy _1, const 42_u64);
    // CHECK: [[cmp]] = const false;
    // CHECK: switchInt(copy [[cmp]]) -> [1: [[BB1:bb.*]], otherwise: [[BB2:bb.*]]];
    // CHECK: [[BB1]]:
    // CHECK: _0 = const 0_i32;
    // CHECK: [[BB2]]:
    // CHECK: _0 = const 1_i32;
    mir! {
        let a: bool;
        {
            a = v == 42;
            a = false;
            match a {
                true => bb1,
                _ => bb2,
            }

        }
        bb1 = {
            RET = 0;
            Return()
        }
        bb2 = {
            RET = 1;
            Return()
        }
    }
}

// EMIT_MIR if_condition_int.on_non_ssa_cmp.SimplifyComparisonIntegral.diff
#[custom_mir(dialect = "runtime")]
pub fn on_non_ssa_cmp(mut v: u64) -> i32 {
    // CHECK-LABEL: fn on_non_ssa_cmp(
    // CHECK: [[cmp:_.*]] = Eq(copy _1, const 42_u64);
    // CHECK: _1 = const 43_u64;
    // CHECK: switchInt(copy [[cmp]]) -> [1: [[BB1:bb.*]], otherwise: [[BB2:bb.*]]];
    // CHECK: [[BB1]]:
    // CHECK: _0 = const 0_i32;
    // CHECK: [[BB2]]:
    // CHECK: _0 = const 1_i32;
    mir! {
        let a: bool;
        {
            a = v == 42;
            v = 43;
            match a {
                true => bb1,
                _ => bb2,
            }

        }
        bb1 = {
            RET = 0;
            Return()
        }
        bb2 = {
            RET = 1;
            Return()
        }
    }
}

// EMIT_MIR if_condition_int.on_non_ssa_place.SimplifyComparisonIntegral.diff
#[custom_mir(dialect = "runtime")]
pub fn on_non_ssa_place(mut v: [u64; 10], mut i: usize) -> i32 {
    // CHECK-LABEL: fn on_non_ssa_place(
    // CHECK: [[cmp:_.*]] = Eq(copy _1[_2], const 42_u64);
    // CHECK: _2 = const 10_usize;
    // CHECK: switchInt(copy [[cmp]]) -> [1: [[BB1:bb.*]], otherwise: [[BB2:bb.*]]];
    // CHECK: [[BB1]]:
    // CHECK: _0 = const 0_i32;
    // CHECK: [[BB2]]:
    // CHECK: _0 = const 1_i32;
    mir! {
        let a: bool;
        {
            a = v[i] == 42;
            i = 10;
            match a {
                true => bb1,
                _ => bb2,
            }

        }
        bb1 = {
            RET = 0;
            Return()
        }
        bb2 = {
            RET = 1;
            Return()
        }
    }
}

fn main() {
    opt_u32(0);
    opt_char('0');
    opt_i8(22);
    dont_opt_bool(false);
    opt_negative(0);
    opt_multiple_ifs(0);
    dont_remove_comparison(11);
    dont_opt_floats(1.0);
    on_non_ssa_switch(42);
}
