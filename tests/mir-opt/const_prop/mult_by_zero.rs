// unit-test: GVN

// EMIT_MIR mult_by_zero.test.GVN.diff
fn test(x: i32) -> i32 {
    // CHECK: fn test(
    // FIXME(cjgillot) simplify algebraic identity
    // CHECK-NOT: _0 = const 0_i32;
    x * 0
}

fn main() {
    test(10);
}
