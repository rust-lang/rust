// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// unit-test: ConstProp

// EMIT_MIR mutable_variable_aggregate_partial_read.main.ConstProp.diff
fn main() {
    let mut x: (i32, i32) = foo();
    x.1 = 99;
    x.0 = 42;
    let y = x.1;
}

#[inline(never)]
fn foo() -> (i32, i32) {
    unimplemented!()
}
