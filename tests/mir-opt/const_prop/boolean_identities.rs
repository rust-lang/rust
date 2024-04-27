//@ test-mir-pass: GVN

// EMIT_MIR boolean_identities.test.GVN.diff
pub fn test(x: bool, y: bool) -> bool {
    // CHECK-LABEL: fn test(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];
    // CHECK: [[a]] = const true;
    // CHECK: [[b]] = const false;
    // CHECK: _0 = const false;
    let a = (y | true);
    let b = (x & false);
    a & b
}

fn main() {
    test(true, false);
}
