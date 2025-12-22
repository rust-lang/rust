//@ test-mir-pass: GVN
//@ compile-flags: -Zinline-mir --crate-type=lib -Cpanic=abort

#![feature(custom_mir, core_intrinsics)]

use std::intrinsics::mir::*;

// EMIT_MIR gvn_range.on_if.GVN.diff
pub fn on_if(i: usize, v: &[u8]) -> u8 {
    // CHECK-LABEL: fn on_if(
    // CHECK: assert(const true
    if i < v.len() { v[i] } else { 0 }
}

// EMIT_MIR gvn_range.on_assert.GVN.diff
pub fn on_assert(i: usize, v: &[u8]) -> u8 {
    // CHECK-LABEL: fn on_assert(
    // CHECK: assert(const true
    assert!(i < v.len());
    v[i]
}

// EMIT_MIR gvn_range.on_match.GVN.diff
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

// EMIT_MIR gvn_range.on_match_2.GVN.diff
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

// EMIT_MIR gvn_range.on_if_2.GVN.diff
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
