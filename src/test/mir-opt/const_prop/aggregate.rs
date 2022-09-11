// unit-test: ConstProp
// compile-flags: -O

// EMIT_MIR aggregate.main.ConstProp.diff
fn main() {
    let x = (0, 1, 2).1 + 0;
}
