// unit-test: ConstProp

// EMIT_MIR mult_by_zero.test.ConstProp.diff
fn test(x: i32) -> i32 {
    // CHECK: fn test(
    // CHECK: _0 = const 0_i32;
    x * 0
}

fn main() {
    test(10);
}
