//@ test-mir-pass: EarlyOtherwiseBranch
//@ compile-flags: -Zmir-enable-passes=+UnreachableEnumBranching

#![feature(custom_mir, core_intrinsics)]

use std::intrinsics::mir::*;

enum Option2<T> {
    Some(T),
    None,
    Other,
}

// We can't optimize it because y may be an invalid value.
// EMIT_MIR early_otherwise_branch.opt1.EarlyOtherwiseBranch.diff
fn opt1(x: Option<u32>, y: Option<u32>) -> u32 {
    // CHECK-LABEL: fn opt1(
    // CHECK: bb0: {
    // CHECK: [[LOCAL1:_.*]] = discriminant({{.*}});
    // CHECK-NOT: Ne
    // CHECK-NOT: discriminant
    // CHECK: switchInt(move [[LOCAL1]]) -> [
    // CHECK-NEXT: }
    match (x, y) {
        (Some(a), Some(b)) => 0,
        _ => 1,
    }
}

// FIXME: `switchInt` will have three targets after `UnreachableEnumBranching`,
// otherwise is unreachable. We can consume the UB fact to transform back to if else pattern.
// EMIT_MIR early_otherwise_branch.opt2.EarlyOtherwiseBranch.diff
fn opt2(x: Option<u32>, y: Option<u32>) -> u32 {
    // CHECK-LABEL: fn opt2(
    // CHECK: bb0: {
    // CHECK: [[LOCAL1:_.*]] = discriminant({{.*}});
    // CHECK-NOT: Ne
    // CHECK-NOT: discriminant
    // CHECK: switchInt(move [[LOCAL1]]) -> [
    // CHECK-NEXT: }
    match (x, y) {
        (Some(a), Some(b)) => 0,
        (None, None) => 2,
        _ => 1,
    }
}

// optimize despite different types
// EMIT_MIR early_otherwise_branch.opt3.EarlyOtherwiseBranch.diff
fn opt3(x: Option2<u32>, y: Option2<bool>) -> u32 {
    // CHECK-LABEL: fn opt3(
    // CHECK: let mut [[CMP_LOCAL:_.*]]: bool;
    // CHECK: bb0: {
    // CHECK: [[LOCAL1:_.*]] = discriminant({{.*}});
    // CHECK: [[LOCAL2:_.*]] = discriminant({{.*}});
    // CHECK: [[CMP_LOCAL]] = Ne(copy [[LOCAL1]], move [[LOCAL2]]);
    // CHECK: switchInt(move [[CMP_LOCAL]]) -> [
    // CHECK-NEXT: }
    match (x, y) {
        (Option2::Some(a), Option2::Some(b)) => 0,
        (Option2::None, Option2::None) => 2,
        (Option2::Other, Option2::Other) => 3,
        _ => 1,
    }
}

// EMIT_MIR early_otherwise_branch.opt4.EarlyOtherwiseBranch.diff
fn opt4(x: Option2<u32>, y: Option2<u32>) -> u32 {
    // CHECK-LABEL: fn opt4(
    // CHECK: let mut [[CMP_LOCAL:_.*]]: bool;
    // CHECK: bb0: {
    // CHECK: [[LOCAL1:_.*]] = discriminant({{.*}});
    // CHECK: [[LOCAL2:_.*]] = discriminant({{.*}});
    // CHECK: [[CMP_LOCAL]] = Ne(copy [[LOCAL1]], move [[LOCAL2]]);
    // CHECK: switchInt(move [[CMP_LOCAL]]) -> [
    // CHECK-NEXT: }
    match (x, y) {
        (Option2::Some(a), Option2::Some(b)) => 0,
        (Option2::None, Option2::None) => 2,
        (Option2::Other, Option2::Other) => 3,
        _ => 1,
    }
}

// EMIT_MIR early_otherwise_branch.opt5.EarlyOtherwiseBranch.diff
fn opt5(x: u32, y: u32) -> u32 {
    // CHECK-LABEL: fn opt5(
    // CHECK: let mut [[CMP_LOCAL:_.*]]: bool;
    // CHECK: bb0: {
    // CHECK: [[CMP_LOCAL]] = Ne(
    // CHECK: switchInt(move [[CMP_LOCAL]]) -> [
    // CHECK-NEXT: }
    match (x, y) {
        (1, 1) => 4,
        (2, 2) => 5,
        (3, 3) => 6,
        _ => 0,
    }
}

// EMIT_MIR early_otherwise_branch.opt5_failed.EarlyOtherwiseBranch.diff
fn opt5_failed(x: u32, y: u32) -> u32 {
    // CHECK-LABEL: fn opt5_failed(
    // CHECK: bb0: {
    // CHECK-NOT: Ne(
    // CHECK: switchInt(
    // CHECK-NEXT: }
    match (x, y) {
        (1, 1) => 4,
        (2, 2) => 5,
        (3, 2) => 6,
        _ => 0,
    }
}

// EMIT_MIR early_otherwise_branch.opt5_failed_type.EarlyOtherwiseBranch.diff
fn opt5_failed_type(x: u32, y: u64) -> u32 {
    // CHECK-LABEL: fn opt5_failed_type(
    // CHECK: bb0: {
    // CHECK-NOT: Ne(
    // CHECK: switchInt(
    // CHECK-NEXT: }
    match (x, y) {
        (1, 1) => 4,
        (2, 2) => 5,
        (3, 3) => 6,
        _ => 0,
    }
}

// EMIT_MIR early_otherwise_branch.target_self.EarlyOtherwiseBranch.diff
#[custom_mir(dialect = "runtime")]
fn target_self(val: i32) {
    // CHECK-LABEL: fn target_self(
    // CHECK: Ne(
    // CHECK-NEXT: switchInt
    mir! {
        {
            Goto(bb1)
        }
        bb1 = {
            match val {
                0 => bb2,
                _ => bb1,
            }
        }
        bb2 = {
            match val {
                0 => bb3,
                _ => bb1,
            }
        }
        bb3 = {
            Return()
        }
    }
}

fn main() {
    opt1(None, Some(0));
    opt2(None, Some(0));
    opt3(Option2::None, Option2::Some(false));
    opt4(Option2::None, Option2::Some(0));
    opt5(0, 0);
    opt5_failed(0, 0);
    target_self(1);
}
