// unit-test: GVN

// EMIT_MIR boolean_identities.test.GVN.diff
pub fn test(x: bool, y: bool) -> bool {
    // CHECK-LABEL: fn test(
    // CHECK: debug a => const true;
    // CHECK: debug b => const false;
    // CHECK: _0 = const false;
    let a = (y | true);
    let b = (x & false);
    a & b
}

fn main() {
    test(true, false);
}
