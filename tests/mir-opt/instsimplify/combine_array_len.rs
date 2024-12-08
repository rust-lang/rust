// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: InstSimplify-after-simplifycfg

// EMIT_MIR combine_array_len.norm2.InstSimplify-after-simplifycfg.diff
fn norm2(x: [f32; 2]) -> f32 {
    // CHECK-LABEL: fn norm2(
    // CHECK-NOT: Len(
    let a = x[0];
    let b = x[1];
    a * a + b * b
}

fn main() {
    assert_eq!(norm2([3.0, 4.0]), 5.0 * 5.0);
}
