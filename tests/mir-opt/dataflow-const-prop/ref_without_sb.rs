// ignore-wasm32 compiled with panic=abort by default
// unit-test: DataflowConstProp

#[inline(never)]
fn escape<T>(x: &T) {}

#[inline(never)]
fn some_function() {}

// EMIT_MIR ref_without_sb.main.DataflowConstProp.diff
fn main() {
    let mut a = 0;
    escape(&a);
    a = 1;
    some_function();
    // This should currently not be propagated.
    let b = a;
}
