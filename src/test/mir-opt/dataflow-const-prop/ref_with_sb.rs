// unit-test: DataflowConstProp
// compile-flags: -Zunsound-mir-opts

#[inline(never)]
fn escape<T>(x: &T) {}

#[inline(never)]
fn some_function() {}

// EMIT_MIR ref_with_sb.main.DataflowConstProp.diff
fn main() {
    let mut a = 0;
    escape(&a);
    a = 1;
    some_function();
    // With `-Zunsound-mir-opt`, this should be propagated
    // (because we assume Stacked Borrows).
    let b = a;
}
