// unit-test: ConstProp
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// EMIT_MIR tuple_literal_propagation.main.ConstProp.diff
fn main() {
    let x = (1, 2);

    consume(x);
}

#[inline(never)]
fn consume(_: (u32, u32)) { }
