//@ test-mir-pass: MatchBranchSimplification

#![crate_type = "lib"]

// Both arms assign the same unit constant with different source spans.
// EMIT_MIR match_branch_simplification_zst.same_unit.MatchBranchSimplification.diff
pub fn same_unit(condition: bool) {
    // CHECK-LABEL: fn same_unit(
    // CHECK-NOT: switchInt
    // CHECK: _0 = no_retag const ();
    // CHECK-NOT: switchInt
    // CHECK: return;
    if condition {}
}
