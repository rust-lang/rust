// skip-filecheck
// unit-test: DataflowConstProp

// EMIT_MIR if.main.DataflowConstProp.diff
fn main() {
    let a = 1;
    let b = if a == 1 { 2 } else { 3 };
    let c = b + 1;

    let d = if a == 1 { a } else { a + 1 };
    let e = d + 1;
}
