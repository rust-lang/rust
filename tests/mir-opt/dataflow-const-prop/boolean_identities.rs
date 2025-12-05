//@ test-mir-pass: DataflowConstProp

// EMIT_MIR boolean_identities.test.DataflowConstProp.diff

// CHECK-LABEL: fn test(
pub fn test(x: bool, y: bool) -> bool {
    // CHECK-NOT: BitAnd(
    // CHECK-NOT: BitOr(
    (y | true) & (x & false)
    // CHECK: _0 = const false;
    // CHECK-NOT: BitAnd(
    // CHECK-NOT: BitOr(
}

// CHECK-LABEL: fn main(
fn main() {
    test(true, false);
}
