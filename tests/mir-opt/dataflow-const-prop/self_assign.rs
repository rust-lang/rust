// skip-filecheck
// unit-test: DataflowConstProp

// EMIT_MIR self_assign.main.DataflowConstProp.diff
fn main() {
    let mut a = 0;
    a = a + 1;
    a = a;

    let mut b = &a;
    b = b;
    a = *b;
}
