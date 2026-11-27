// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: GVN

// This used to ICE in const-prop

fn test(this: ((u8, u8),)) {
    assert!((this.0).0 == 1);
}

// EMIT_MIR issue_67019.main.GVN.diff
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: = test(const ((1_u8, 2_u8),))
    test(((1, 2),));
}
