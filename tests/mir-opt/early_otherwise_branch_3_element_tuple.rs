//@ test-mir-pass: EarlyOtherwiseBranch
//@ compile-flags: -Zmir-enable-passes=+UnreachableEnumBranching

enum Option2<T> {
    Some(T),
    None,
    Other,
}

// FIXME: `switchInt` will have three targets after `UnreachableEnumBranching`,
// otherwise is unreachable. We can consume the UB fact to transform back to if else pattern.
// EMIT_MIR early_otherwise_branch_3_element_tuple.opt1.EarlyOtherwiseBranch.diff
fn opt1(x: Option<u32>, y: Option<u32>, z: Option<u32>) -> u32 {
    // CHECK-LABEL: fn opt1(
    // CHECK: bb0: {
    // CHECK: [[LOCAL1:_.*]] = discriminant({{.*}});
    // CHECK-NOT: Ne
    // CHECK-NOT: discriminant
    // CHECK: switchInt(move [[LOCAL1]]) -> [
    // CHECK-NEXT: }
    match (x, y, z) {
        (Some(a), Some(b), Some(c)) => 0,
        (None, None, None) => 2,
        _ => 1,
    }
}

// EMIT_MIR early_otherwise_branch_3_element_tuple.opt2.EarlyOtherwiseBranch.diff
fn opt2(x: Option2<u32>, y: Option2<u32>, z: Option2<u32>) -> u32 {
    // CHECK-LABEL: fn opt2(
    // CHECK: let mut [[CMP_LOCAL:_.*]]: bool;
    // CHECK: bb0: {
    // CHECK: [[LOCAL1:_.*]] = discriminant({{.*}});
    // CHECK: [[LOCAL2:_.*]] = discriminant({{.*}});
    // CHECK: [[CMP_LOCAL]] = Ne(copy [[LOCAL1]], move [[LOCAL2]]);
    // CHECK: switchInt(move [[CMP_LOCAL]]) -> [
    // CHECK-NEXT: }
    match (x, y, z) {
        (Option2::Some(a), Option2::Some(b), Option2::Some(c)) => 0,
        (Option2::None, Option2::None, Option2::None) => 2,
        (Option2::Other, Option2::Other, Option2::Other) => 3,
        _ => 1,
    }
}

fn main() {
    opt1(None, Some(0), None);
    opt2(Option2::None, Option2::Some(0), Option2::None);
}
