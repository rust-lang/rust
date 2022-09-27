// unit-test: DataflowConstProp

// EMIT_MIR mutable_variable_aggregate_partial_read.main.DataflowConstProp.diff
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
