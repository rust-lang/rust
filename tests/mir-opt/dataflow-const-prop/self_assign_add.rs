//@ test-mir-pass: DataflowConstProp

// EMIT_MIR self_assign_add.main.DataflowConstProp.diff

// CHECK-LABEL: fn main(
fn main() {
    // CHECK: debug a => [[a:_.*]];
    let mut a = 0;

    // CHECK: [[a]] = const 1_i32;
    a += 1;

    // CHECK: [[a]] = const 2_i32;
    a += 1;
}
