// unit-test: ConstProp
// EMIT_MIR ref_deref.main.ConstProp.diff

fn main() {
    *(&4);
}
