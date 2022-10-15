// compile-flags: -C overflow-checks=off

// EMIT_MIR inherit_overflow_checks_use.main.DataflowConstProp.diff
fn main() {
    // After inlining, this will contain a `CheckedBinaryOp`. The overflow
    // must be ignored by the constant propagation to avoid triggering a panic.
    let _ = <u8 as std::ops::Add>::add(255, 1);
}
