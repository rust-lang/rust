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
    // Without `-Zunsound-mir-opt`, this should not be propagated
    // (because we do not assume Stacked Borrows).
    let b = a;
}
