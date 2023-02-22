// unit-test: DataflowConstProp
// compile-flags: -Zmir-enable-passes=+Inline

// EMIT_MIR inherit_overflow.main.DataflowConstProp.diff
fn main() {
    // After inlining, this will contain a `CheckedBinaryOp`.
    // Propagating the overflow is ok as codegen will just skip emitting the panic.
    let _ = <u8 as std::ops::Add>::add(255, 1);
}
