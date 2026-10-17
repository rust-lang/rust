//@ test-mir-pass: DataflowConstProp

// The struct has scalar ABI, but is not a scalar type.
// Make sure that we handle this correctly.
#[repr(transparent)]
struct I32(i32);

// EMIT_MIR repr_transparent.main.DataflowConstProp.diff

// CHECK-LABEL: fn main(
fn main() {
    // CHECK: debug x => [[x:_.*]];
    // CHECK: debug y => [[y:_.*]];

    // CHECK: [[x]] = const I32(0_i32);
    let x = I32(0);

    // CHECK: [[y]] = const I32(0_i32);
    let y = I32(x.0 + x.0);
}
