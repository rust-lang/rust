// skip-filecheck
// unit-test: ConstProp

// EMIT_MIR self_assign_add.main.ConstProp.diff
fn main() {
    let mut a = 0;
    a += 1;
    a += 1;
}
