//@ test-mir-pass: SimplifyConstCondition-after-inst-simplify
//@ compile-flags: -Zmir-enable-passes=+InstSimplify-after-simplifycfg -Zub_checks=false -Zinline-mir

#![crate_type = "lib"]

// EMIT_MIR trivial_const.unwrap_unchecked.SimplifyConstCondition-after-inst-simplify.diff
pub fn unwrap_unchecked(v: &Option<i32>) -> i32 {
    // CHECK-LABEL: fn unwrap_unchecked(
    // CHECK: bb0: {
    // CHECK: switchInt({{.*}}) -> [0: [[AssumeFalseBB:bb.*]], 1:
    // CHECK: [[AssumeFalseBB]]: {
    // CHECK-NEXT: unreachable;
    let v = unsafe { v.unwrap_unchecked() };
    v
}
