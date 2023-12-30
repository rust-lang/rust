// unit-test: GVN

// EMIT_MIR boolean_identities.test.GVN.diff
pub fn test(x: bool, y: bool) -> bool {
    // CHECK-LABEL: fn test(
    // CHECK: debug a => [[a:_.*]];
    // CHECK: debug b => [[b:_.*]];
    // FIXME(cjgillot) simplify algebraic identity
    // CHECK-NOT: [[a]] = const true;
    // CHECK-NOT: [[b]] = const false;
    // CHECK-NOT: _0 = const false;
    let a = (y | true);
    let b = (x & false);
    a & b
}

fn main() {
    test(true, false);
}
