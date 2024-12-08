//@ test-mir-pass: InstSimplify-after-simplifycfg
//@ compile-flags: -Cdebug-assertions=no -Zinline-mir

// EMIT_MIR ub_check.unwrap_unchecked.InstSimplify-after-simplifycfg.diff
pub fn unwrap_unchecked(x: Option<i32>) -> i32 {
    // CHECK-LABEL: fn unwrap_unchecked(
    // CHECK-NOT: UbChecks()
    // CHECK: [[assume:_.*]] = const false;
    // CHECK-NEXT: assume(copy [[assume]]);
    // CHECK-NEXT: unreachable_unchecked::precondition_check
    unsafe { x.unwrap_unchecked() }
}

fn main() {
    unwrap_unchecked(None);
}
