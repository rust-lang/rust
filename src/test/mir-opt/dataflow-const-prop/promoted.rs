// unit-test: DataflowConstProp
// compile-flags: -Zunsound-mir-opts

// EMIT_MIR promoted.main.DataflowConstProp.diff
fn main() {
    // This does not work because `&42` gets promoted.
    let a = *&42;
}
