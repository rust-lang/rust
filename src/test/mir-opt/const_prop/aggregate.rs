// unit-test: ConstProp
// compile-flags: -O

// EMIT_MIR aggregate.main.ConstProp.diff
// EMIT_MIR aggregate.main.PreCodegen.after.mir
fn main() {
    let x = (0, 1, 2).1 + 0;
}
