//@ test-mir-pass: DataflowConstProp
//@ compile-flags: -Zmir-enable-passes=+GVN,+Inline -Zdump-mir-exclude-alloc-bytes
// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

// This test is to check ICE in issue [#115789](https://github.com/rust-lang/rust/issues/115789).

struct A {
    foo: Box<[bool]>,
}

// EMIT_MIR default_boxed_slice.main.GVN.diff
// EMIT_MIR default_boxed_slice.main.DataflowConstProp.diff

// CHECK-LABEL: fn main(
fn main() {
    // ConstProp will create a constant of type `Box<[bool]>`.
    // Verify that `DataflowConstProp` does not ICE trying to dereference it directly.

    // CHECK: debug a => [[a:_.*]];
    // We may check other inlined functions as well...

    // CHECK: {{_.*}} = const Box::<[bool]>(
    let a: A = A { foo: Box::default() };
}
