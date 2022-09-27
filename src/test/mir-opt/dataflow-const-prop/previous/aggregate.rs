// unit-test: DataflowConstProp
// compile-flags: -O

// EMIT_MIR aggregate.main.DataflowConstProp.diff
fn main() {
    let x = (0, 1, 2).1 + 0;
}
