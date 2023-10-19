// skip-filecheck
// unit-test: DataflowConstProp

// EMIT_MIR self_assign_add.main.DataflowConstProp.diff
fn main() {
    let mut a = 0;
    a += 1;
    a += 1;
}
