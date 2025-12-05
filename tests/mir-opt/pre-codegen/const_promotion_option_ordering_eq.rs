//@ compile-flags: -O -Zmir-opt-level=2 -Cdebuginfo=0

// Check that comparing `Option<Ordering>` to a constant inlined `Some(...)`
// does not produce unnecessarily complex MIR compared to using a local binding.
//
// Regression test for <https://github.com/rust-lang/rust/issues/139093>.
// Originally, inlined constants like `Some(Ordering::Equal)` would get promoted,
// leading to more MIR (and extra LLVM IR checks) than necessary.
// Both cases should now generate identical MIR.

use std::cmp::Ordering;

// EMIT_MIR const_promotion_option_ordering_eq.direct.PreCodegen.after.mir
pub fn direct(e: Option<Ordering>) -> bool {
    // CHECK-LABEL: fn direct(
    // CHECK-NOT: promoted[
    // CHECK: switchInt(
    // CHECK: return
    e == Some(Ordering::Equal)
}

// EMIT_MIR const_promotion_option_ordering_eq.with_let.PreCodegen.after.mir
pub fn with_let(e: Option<Ordering>) -> bool {
    // CHECK-LABEL: fn with_let(
    // CHECK-NOT: promoted[
    // CHECK: switchInt(
    // CHECK: return
    let eq = Ordering::Equal;
    e == Some(eq)
}
