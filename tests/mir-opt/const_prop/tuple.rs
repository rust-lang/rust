// skip-filecheck
// unit-test: ConstProp
// EMIT_MIR_FOR_EACH_BIT_WIDTH

// EMIT_MIR tuple.main.ConstProp.diff
fn main() {
    let mut a = (1, 2);
    let b = a.0 + a.1 + 3;
    a = (2, 3);
    let c = a.0 + a.1 + b;

    let d = (b, a, c);
}
