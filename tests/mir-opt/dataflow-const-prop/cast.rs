//@ test-mir-pass: DataflowConstProp

// EMIT_MIR cast.main.DataflowConstProp.diff

// CHECK-LABEL: fn main(
fn main() {
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];

    // CHECK: [[a]] = const 257_i32;
    let a = 257;
    // CHECK: [[b]] = const 2_u8;
    let b = a as u8 + 1;
}
