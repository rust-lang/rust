// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: InstSimplify-after-simplifycfg

// EMIT_MIR combine_array_len.norm2.InstSimplify-after-simplifycfg.diff
fn norm2(x: [f32; 2]) -> f32 {
    // CHECK-LABEL: fn norm2(
    // CHECK-NOT: PtrMetadata(
    let a = x[0];
    let b = x[1];
    a * a + b * b
}

// EMIT_MIR combine_array_len.normN.InstSimplify-after-simplifycfg.diff
fn normN<const N: usize>(x: [f32; N]) -> f32 {
    // CHECK-LABEL: fn normN(
    // CHECK-NOT: PtrMetadata(
    let a = x[0];
    let b = x[1];
    a * a + b * b
}

fn main() {
    assert_eq!(norm2([3.0, 4.0]), 5.0 * 5.0);
    assert_eq!(normN([3.0, 4.0]), 5.0 * 5.0);
}
