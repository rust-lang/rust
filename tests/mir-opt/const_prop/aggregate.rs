// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: ConstProp
// compile-flags: -O

// EMIT_MIR aggregate.main.ConstProp.diff
// EMIT_MIR aggregate.main.PreCodegen.after.mir
fn main() {
    let x = (0, 1, 2).1 + 0;
    foo(x);
}

// EMIT_MIR aggregate.foo.ConstProp.diff
// EMIT_MIR aggregate.foo.PreCodegen.after.mir
fn foo(x: u8) {
    // Verify that we still propagate if part of the aggregate is not known.
    let first = (0, x).0 + 1;
    let second = (x, 1).1 + 2;
}
