// unit-test: DataflowConstProp

// EMIT_MIR ref.main.DataflowConstProp.diff
fn main() {
    let a = 0;
    let b = 0;
    let c = if std::process::id() % 2 == 0 { &a } else { &b };
    let d = *c + 1;
}
