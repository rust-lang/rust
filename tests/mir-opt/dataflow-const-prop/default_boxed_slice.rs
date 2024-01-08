// unit-test: DataflowConstProp
// compile-flags: -Zmir-enable-passes=+GVN,+Inline
// ignore-debug assertions change the output MIR
// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

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
    // CHECK: scope {{[0-9]+}} (inlined <Box<[bool]> as Default>::default) {
    // CHECK: scope {{[0-9]+}} (inlined Unique::<[bool; 0]>::dangling) {
    // CHECK: scope {{[0-9]+}} (inlined NonNull::<[bool; 0]>::dangling) {
    // We may check other inlined functions as well...

    // CHECK: bb{{[0-9]+}}: {
    // CHECK: [[box_obj:_.*]] = Box::<[bool]>(_3, const std::alloc::Global);
    // CHECK: [[a]] = A { foo: move [[box_obj]] };
    let a: A = A { foo: Box::default() };
}
