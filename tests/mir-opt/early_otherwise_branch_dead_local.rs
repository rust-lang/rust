//@ test-mir-pass: EarlyOtherwiseBranch
//@ compile-flags: -Zlint-mir=no

#![feature(core_intrinsics, custom_mir)]

use std::intrinsics::mir::*;

fn main() {
    dead_local(1);
    constant_parent_dead_local();
}

// EMIT_MIR early_otherwise_branch_dead_local.dead_local.EarlyOtherwiseBranch.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
#[inline(never)]
fn dead_local(a: u64) {
    // CHECK-LABEL: fn dead_local(
    // CHECK-SAME: [[A:_.*]]: u64) -> () {
    // CHECK: let mut [[B:_.*]]: u64;
    // CHECK: bb0: {
    // CHECK-NEXT: StorageLive([[B]]);
    // CHECK-NEXT: [[B]] = const 0_u64;
    // CHECK-NEXT: StorageDead([[B]]);
    // CHECK-NEXT: switchInt(copy [[A]]) -> [0: bb{{[0-9]+}}, otherwise: bb{{[0-9]+}}];
    // CHECK-NEXT: }
    mir! {
        let b: u64;
        let marker: u64;

        {
            StorageLive(b);
            b = 0;
            StorageDead(b);
            match a {
                0 => bb1,
                _ => bb3,
            }
        }

        bb1 = {
            match b {
                0 => bb2,
                _ => bb3,
            }
        }

        bb2 = {
            marker = 2;
            Goto(bb4)
        }

        bb3 = {
            marker = 3;
            Goto(bb4)
        }

        bb4 = {
            Return()
        }
    }
}

// EMIT_MIR early_otherwise_branch_dead_local.constant_parent_dead_local.EarlyOtherwiseBranch.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
fn constant_parent_dead_local() {
    // CHECK-LABEL: fn constant_parent_dead_local(
    // CHECK: let mut [[B:_.*]]: u64;
    // CHECK: bb0: {
    // CHECK-NEXT: StorageLive([[B]]);
    // CHECK-NEXT: [[B]] = const 0_u64;
    // CHECK-NEXT: StorageDead([[B]]);
    // CHECK-NEXT: switchInt(const 1_u64) -> [0: bb{{[0-9]+}}, otherwise: bb{{[0-9]+}}];
    // CHECK-NEXT: }
    mir! {
        let b: u64;
        let marker: u64;

        {
            StorageLive(b);
            b = 0;
            StorageDead(b);
            match 1_u64 {
                0 => bb1,
                _ => bb3,
            }
        }

        bb1 = {
            match b {
                0 => bb2,
                _ => bb3,
            }
        }

        bb2 = {
            marker = 2;
            Goto(bb4)
        }

        bb3 = {
            marker = 3;
            Goto(bb4)
        }

        bb4 = {
            Return()
        }
    }
}
