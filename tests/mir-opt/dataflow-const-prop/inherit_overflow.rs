// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: DataflowConstProp
//@ compile-flags: -Zmir-enable-passes=+Inline

// EMIT_MIR inherit_overflow.main.DataflowConstProp.diff
// CHECK-LABEL: fn main(
fn main() {
    // After inlining, this will contain a `CheckedBinaryOp`.
    // Propagating the overflow is ok as codegen will just skip emitting the panic.

    // CHECK: {{_.*}} = const (0_u8, true);
    // CHECK: assert(!const true,
    let _ = <u8 as std::ops::Add>::add(255, 1);
}
