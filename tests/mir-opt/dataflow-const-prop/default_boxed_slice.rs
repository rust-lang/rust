// skip-filecheck
// unit-test: DataflowConstProp
// compile-flags: -Zmir-enable-passes=+ConstProp,+Inline
// ignore-debug assertions change the output MIR
// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

struct A {
    foo: Box<[bool]>,
}

// EMIT_MIR default_boxed_slice.main.ConstProp.diff
// EMIT_MIR default_boxed_slice.main.DataflowConstProp.diff
fn main() {
    // ConstProp will create a constant of type `Box<[bool]>`.
    // Verify that `DataflowConstProp` does not ICE trying to dereference it directly.
    let a: A = A { foo: Box::default() };
}
