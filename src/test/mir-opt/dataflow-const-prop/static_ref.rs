// unit-test: DataflowConstProp

// EMIT_MIR static_ref.main.DataflowConstProp.diff
fn main() {
    static P: usize = 5;
    let x = 0;
    let mut r = &x;
    r = &P;
    let y = *r;
}
