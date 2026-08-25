//@ test-mir-pass: SimplifyCfg-after-unreachable-enum-branching
#![feature(custom_mir, core_intrinsics)]
#![crate_type = "lib"]

use std::intrinsics::mir::*;

// Check that we correctly cleaned up the dead BB.
// EMIT_MIR simplify_dead_blocks.assert_nonzero_nonmax.SimplifyCfg-after-unreachable-enum-branching.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub unsafe fn assert_nonzero_nonmax(x: u8) -> u8 {
    // CHECK-LABEL: fn assert_nonzero_nonmax(
    // CHECK: bb0: {
    // CHECK-NEXT: switchInt(copy {{_[0-9]+}}) -> [0: [[unreachable:bb.*]], 1: [[retblock2:bb.*]], 255: [[unreachable:bb.*]], otherwise: [[retblock:bb.*]]];
    // CHECK-NEXT: }
    // CHECK-NOT: _0 = const 1_u8;
    // CHECK: [[retblock2]]: {
    // CHECK-NEXT: _0 = const 2_u8;
    // CHECK-NEXT: return;
    // CHECK-NEXT: }
    // CHECK: [[unreachable]]: {
    // CHECK-NEXT: unreachable;
    // CHECK-NEXT: }
    // CHECK: [[retblock]]: {
    // CHECK-NEXT: _0 = copy _1;
    // CHECK-NEXT: return;
    // CHECK-NEXT: }
    mir! {
        {
            match x {
                0 => unreachable,
                1 => retblock2,
                u8::MAX => unreachable,
                _ => retblock,
            }
        }
        deadRetblock1 = {
            RET = 1;
            Return()
        }
        retblock2 = {
            RET = 2;
            Return()
        }
        unreachable = {
            Unreachable()
        }
        retblock = {
            RET = x;
            Return()
        }
    }
}
