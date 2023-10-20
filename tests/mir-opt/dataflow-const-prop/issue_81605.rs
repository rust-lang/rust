// skip-filecheck
// unit-test: DataflowConstProp

// EMIT_MIR issue_81605.f.DataflowConstProp.diff
fn f() -> usize {
    1 + if true { 1 } else { 2 }
}

fn main() {
    f();
}
