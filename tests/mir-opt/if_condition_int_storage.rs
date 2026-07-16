//@ test-mir-pass: SimplifyComparisonIntegral
// MIR lint assumes all paths can execute. The backedge is undefined, but the first traversal is
// defined and must stay defined.
//@ compile-flags: -Zlint-mir=false

#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

// EMIT_MIR if_condition_int_storage.dont_opt_storage_dead_in_loop.SimplifyComparisonIntegral.diff
// Regression test for https://github.com/rust-lang/rust/issues/158231.
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn dont_opt_storage_dead_in_loop(input: u32) {
    // CHECK-LABEL: fn dont_opt_storage_dead_in_loop(
    // CHECK: [[a:_.*]] = copy _1;
    // CHECK: [[cmp:_.*]] = Eq(copy [[a]], const 123_u32);
    // CHECK: StorageDead([[a]]);
    // CHECK: switchInt(move [[cmp]]) -> [1: [[BB1:bb.*]], otherwise: [[BB2:bb.*]]];
    mir! {
        let a: u32;
        let cmp: bool;
        {
            StorageLive(a);
            a = input;
            Goto(bb1)
        }
        bb1 = {
            Goto(bb2)
        }
        bb2 = {
            cmp = a == 123;
            StorageDead(a);
            match Move(cmp) {
                true => bb1,
                _ => bb3,
            }
        }
        bb3 = {
            Return()
        }
    }
}

fn main() {}
