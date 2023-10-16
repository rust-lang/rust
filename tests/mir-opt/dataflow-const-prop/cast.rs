// skip-filecheck
// unit-test: DataflowConstProp

// EMIT_MIR cast.main.DataflowConstProp.diff
fn main() {
    let a = 257;
    let b = a as u8 + 1;
}
