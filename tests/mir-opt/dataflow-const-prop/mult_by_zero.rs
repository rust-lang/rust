//@ test-mir-pass: DataflowConstProp

// EMIT_MIR mult_by_zero.test.DataflowConstProp.diff
// CHECK-LABEL: fn test(
fn test(x: i32) -> i32 {
    x * 0
    // CHECK: _0 = const 0_i32;
}

fn main() {
    test(10);
}
