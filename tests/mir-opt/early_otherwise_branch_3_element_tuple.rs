//@ unit-test: EarlyOtherwiseBranch
//@ compile-flags: -Zmir-enable-passes=+UninhabitedEnumBranching

// EMIT_MIR early_otherwise_branch_3_element_tuple.opt1.EarlyOtherwiseBranch.diff
fn opt1(x: Option<u32>, y: Option<u32>, z: Option<u32>) -> u32 {
    // CHECK-LABEL: fn opt1(
    // CHECK: let mut [[CMP_LOCAL:_.*]]: bool;
    // CHECK: bb0: {
    // CHECK: [[LOCAL1:_.*]] = discriminant({{.*}});
    // CHECK: [[LOCAL2:_.*]] = discriminant({{.*}});
    // CHECK: [[CMP_LOCAL]] = Ne([[LOCAL1]], move [[LOCAL2]]);
    // CHECK: switchInt(move [[CMP_LOCAL]]) -> [
    // CHECK-NEXT: }
    match (x, y, z) {
        (Some(a), Some(b), Some(c)) => 0,
        (None, None, None) => 0,
        _ => 1,
    }
}

fn main() {
    opt1(None, Some(0), None);
}
