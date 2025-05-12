// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: GVN
//@ compile-flags: -Zmir-enable-passes=+Inline

// After inlining, this will contain a `CheckedBinaryOp`.
// Propagating the overflow is ok as codegen will just skip emitting the panic.
// EMIT_MIR inherit_overflow.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: {{_.*}} = const (0_u8, true);
    // CHECK: assert(!const true,
    // CHECK: {{_.*}} = const 0_u8;
    let _ = <u8 as std::ops::Add>::add(255, 1);
}
