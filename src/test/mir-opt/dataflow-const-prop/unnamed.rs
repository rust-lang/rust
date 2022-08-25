// unit-test: DataflowConstProp

static g: i32 = 2;

// EMIT_MIR unnamed.main.DataflowConstProp.diff
fn main() {
    let mut a = 0;
    a += 1;
    a += g;
}
