// skip-filecheck
// unit-test: ConstProp

// EMIT_MIR issue_81605.f.ConstProp.diff
fn f() -> usize {
    1 + if true { 1 } else { 2 }
}

fn main() {
    f();
}
