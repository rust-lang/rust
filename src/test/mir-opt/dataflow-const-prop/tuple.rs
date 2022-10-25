// unit-test: DataflowConstProp
// compile-flags: -Zunsound-mir-opts

// EMIT_MIR tuple.main.DataflowConstProp.diff
fn main() {
    let mut a = (1, 2);
    let mut b = &a;
    let c = a.0 + b.1 + 3;
    a = (2, 3);
    b = &a;
    let d = a.0 + b.1 + 4;
}
