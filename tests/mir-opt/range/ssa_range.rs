//@ test-mir-pass: SsaRangePropagation
//@ compile-flags: -Zmir-enable-passes=+GVN,+Inline --crate-type=lib -Cpanic=abort

#![feature(custom_mir, core_intrinsics)]

use std::intrinsics::mir::*;

// EMIT_MIR ssa_range.on_if.SsaRangePropagation.diff
pub fn on_if(i: usize, v: &[u8]) -> u8 {
    // CHECK-LABEL: fn on_if(
    // CHECK: assert(const true
    if i < v.len() { v[i] } else { 0 }
}

// EMIT_MIR ssa_range.on_assert.SsaRangePropagation.diff
pub fn on_assert(i: usize, v: &[u8]) -> u8 {
    // CHECK-LABEL: fn on_assert(
    // CHECK: assert(const true
    assert!(i < v.len());
    v[i]
}

// EMIT_MIR ssa_range.on_assume.SsaRangePropagation.diff
pub fn on_assume(i: usize, v: &[u8]) -> u8 {
    // CHECK-LABEL: fn on_assume(
    // CHECK: assert(const true
    unsafe {
        std::intrinsics::assume(i < v.len());
    }
    v[i]
}

// EMIT_MIR ssa_range.on_match.SsaRangePropagation.diff
pub fn on_match(i: u8) -> u8 {
    // CHECK-LABEL: fn on_match(
    // CHECK: switchInt(copy _1) -> [1: [[BB_V1:bb.*]], 2: [[BB_V2:bb.*]],
    // CHECK: [[BB_V2]]: {
    // CHECK-NEXT: _0 = const 2_u8;
    // CHECK: [[BB_V1]]: {
    // CHECK-NEXT: _0 = const 1_u8;
    match i {
        1 => i,
        2 => i,
        _ => 0,
    }
}

// EMIT_MIR ssa_range.on_match_2.SsaRangePropagation.diff
pub fn on_match_2(i: u8) -> u8 {
    // CHECK-LABEL: fn on_match_2(
    // CHECK: switchInt(copy _1) -> [1: [[BB:bb.*]], 2: [[BB]],
    // CHECK: [[BB]]: {
    // CHECK-NEXT: _0 = copy _1;
    match i {
        1 | 2 => i,
        _ => 0,
    }
}

// EMIT_MIR ssa_range.intersect_comparisons.SsaRangePropagation.diff
pub fn intersect_comparisons(x: u8) -> u8 {
    // CHECK-LABEL: fn intersect_comparisons(
    // CHECK: _0 = const 9_u8;
    if x < 10 {
        if x >= 9 {
            return x;
        }
    }
    0
}

// The comparisons below use custom MIR to preserve operand order and avoid
// earlier optimizations hiding the behavior being tested.
// EMIT_MIR ssa_range.swapped_otherwise.SsaRangePropagation.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn swapped_otherwise(x: u8) -> u8 {
    // CHECK-LABEL: fn swapped_otherwise(
    // CHECK: _0 = const 9_u8;
    mir! {
        let low: bool;
        let high: bool;
        {
            low = 9_u8 < x;
            match low { true => fallback, _ => lower }
        }
        lower = {
            high = 9_u8 > x;
            match high { true => fallback, _ => found }
        }
        found = { RET = x; Return() }
        fallback = { RET = 0_u8; Return() }
    }
}

// EMIT_MIR ssa_range.folded_bound.SsaRangePropagation.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn folded_bound(x: u8, bound: u8) -> u8 {
    // CHECK-LABEL: fn folded_bound(
    // CHECK: _0 = const 9_u8;
    mir! {
        let equal: bool;
        {
            match bound { 9 => compare, _ => fallback }
        }
        compare = {
            equal = x == bound;
            match equal { true => found, _ => fallback }
        }
        found = { RET = x; Return() }
        fallback = { RET = 0_u8; Return() }
    }
}

// EMIT_MIR ssa_range.not_equal_otherwise.SsaRangePropagation.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn not_equal_otherwise(x: u8) -> u8 {
    // CHECK-LABEL: fn not_equal_otherwise(
    // CHECK: _0 = const u8::MAX;
    mir! {
        let different: bool;
        {
            different = 255_u8 != x;
            match different { true => fallback, _ => found }
        }
        found = { RET = x; Return() }
        fallback = { RET = 0_u8; Return() }
    }
}

// EMIT_MIR ssa_range.unsigned_boundaries.SsaRangePropagation.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn unsigned_boundaries(x: u8) -> u8 {
    // CHECK-LABEL: fn unsigned_boundaries(
    // CHECK: _0 = const 0_u8;
    // CHECK: _0 = const u8::MAX;
    mir! {
        let at_min: bool;
        let at_max: bool;
        {
            at_min = x <= 0_u8;
            match at_min { true => minimum, _ => upper }
        }
        minimum = { RET = x; Return() }
        upper = {
            at_max = x >= 255_u8;
            match at_max { true => maximum, _ => fallback }
        }
        maximum = { RET = x; Return() }
        fallback = { RET = 1_u8; Return() }
    }
}

// EMIT_MIR ssa_range.signed_comparison.SsaRangePropagation.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn signed_comparison(x: i8) -> i8 {
    // CHECK-LABEL: fn signed_comparison(
    // CHECK: _0 = copy _1;
    mir! {
        let negative: bool;
        {
            negative = x < 0_i8;
            match negative { true => found, _ => fallback }
        }
        found = { RET = x; Return() }
        fallback = { RET = 0_i8; Return() }
    }
}

// EMIT_MIR ssa_range.on_if_2.SsaRangePropagation.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn on_if_2(a: bool) -> bool {
    // CHECK-LABEL: fn on_if_2(
    // CHECK: _0 = copy _1;
    mir! {
        {
            match a {
                true => bb2,
                _ => bb1
            }
        }
        bb1 = {
            Goto(bb2)
        }
        bb2 = {
            RET = a;
            Return()
        }
    }
}
