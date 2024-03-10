//@ unit-test: InstSimplify
//@ compile-flags: -Cdebug-assertions=no -Zinline-mir

// EMIT_MIR ub_check.unwrap_unchecked.InstSimplify.diff
pub fn unwrap_unchecked(x: Option<i32>) -> i32 {
    // CHECK-LABEL: fn unwrap_unchecked(
    // CHECK-NOT: UbCheck(LanguageUb)
    // CHECK: [[assume:_.*]] = const false;
    // CHECK-NEXT: assume([[assume]]);
    // CHECK-NEXT: unreachable_unchecked::precondition_check
    unsafe { x.unwrap_unchecked() }
}

fn main() {
    unwrap_unchecked(None);
}
