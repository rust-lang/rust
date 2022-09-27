// unit-test: DataflowConstProp
// EMIT_MIR tuple_literal_propagation.main.DataflowConstProp.diff
fn main() {
    let x = (1, 2);

    consume(x);
}

#[inline(never)]
fn consume(_: (u32, u32)) {}
