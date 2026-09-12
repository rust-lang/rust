//@ test-mir-pass: UnreachableEnumBranching

enum Empty {}

enum S {
    A(Empty),
    B,
    C,
}

use S::*;

// EMIT_MIR uninhabited_fallthrough_elimination.keep_fallthrough.UnreachableEnumBranching.diff
fn keep_fallthrough(s: S) -> u32 {
    // CHECK-LABEL: fn keep_fallthrough(
    // CHECK: debug s => [[s:_.*]];
    // CHECK: [[DISCR:_.*]] = discriminant([[s]]);
    // CHECK: switchInt({{.*}} [[DISCR]]) -> [
    //
    // CHECK-SAME: 0: [[UNREACH:bb[0-9]+]],
    // CHECK-SAME: 1: [[BB_B:bb[0-9]+]],
    // CHECK-SAME: 2: [[BB_ANY:bb[0-9]+]],
    // CHECK-SAME: otherwise: [[UNREACH]]
    // CHECK-SAME: ];
    //
    // CHECK-DAG: [[BB_B]]: {{{[[:space:]]}} _0 = const 2_u32;
    // CHECK-DAG: [[BB_ANY]]: {{{[[:space:]]}} _0 = const 3_u32;
    // CHECK-DAG: [[UNREACH]]: {{{[[:space:]]}} unreachable;
    match s {
        A(_) => 1,
        B => 2,
        _ => 3,
    }
}

// EMIT_MIR uninhabited_fallthrough_elimination.eliminate_fallthrough.UnreachableEnumBranching.diff
fn eliminate_fallthrough(s: S) -> u32 {
    // CHECK-LABEL: fn eliminate_fallthrough(
    // CHECK debug s => [[s:_.*]];
    // CHECK: [[DISCR:_.*]] = discriminant([[s]]);
    //
    // CHECK: switchInt({{.*}} [[DISCR]]) -> [
    // CHECK-SAME: 1: [[BB_B:bb[0-9]+]],
    // CHECK-SAME: 2: [[BB_C:bb[0-9]+]],
    // CHECK-SAME: otherwise: [[UNREACH:bb[0-9]+]]
    // CHECK-SAME: ];
    //
    // CHECK-DAG: [[BB_B]]: {{{[[:space:]]}} _0 = const 2_u32;
    // CHECK-DAG: [[BB_C]]: {{{[[:space:]]}} _0 = const 1_u32;
    // CHECK-DAG: [[UNREACH]]: {{{[[:space:]]}} unreachable;
    match s {
        C => 1,
        B => 2,
        _ => 3,
    }
}

fn main() {
    keep_fallthrough(B);
    eliminate_fallthrough(B);
}
