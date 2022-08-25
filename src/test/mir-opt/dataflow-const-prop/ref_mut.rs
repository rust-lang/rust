// unit-test: DataflowConstProp

// EMIT_MIR ref_mut.main.DataflowConstProp.diff
fn main() {
    let mut a = 0;
    let b = &mut a;
    *b = 1;
    let c = a;

    let d = 0;
    let mut e = &d;
    let f = &mut e;
    *f = &1;
    let g = *e;
}
