// unit-test: DataflowConstProp
// compile-flags: -Zunsound-mir-opts

// EMIT_MIR static_ref.main.DataflowConstProp.diff
fn main() {
    // Currently, this will not propagate.
    static P: i32 = 5;
    let x = 0;
    let mut r = &x;
    r = &P;
    let y = *r;
}
