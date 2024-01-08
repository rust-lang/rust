// unit-test: DataflowConstProp

// EMIT_MIR boolean_identities.test.DataflowConstProp.diff

// CHECK-LABEL: fn test(
pub fn test(x: bool, y: bool) -> bool {
    (y | true) & (x & false)
    // CHECK: _0 = const false;
}

// CHECK-LABEL: fn main(
fn main() {
    test(true, false);
}
