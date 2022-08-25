// unit-test: DataflowConstProp

// EMIT_MIR promoted.main.DataflowConstProp.diff
fn main() {
    // This does not work because `&42` gets promoted.
    let a = *&42;
}
