// unit-test: DataflowConstProp

#[inline(never)]
fn escape<T>(x: &T) {}

// EMIT_MIR ref_without_sb.main.DataflowConstProp.diff
fn main() {
    let mut a = 0;
    escape(&a);
    a = 1;
    // Without `-Zunsound-mir-opt`, this should not be propagated
    // (because we do not assume Stacked Borrows).
    let b = a;
}
