// ignore-wasm32 compiled with panic=abort by default
// unit-test: InstSimplify
// EMIT_MIR combine_array_len.norm2.InstSimplify.diff

fn norm2(x: [f32; 2]) -> f32 {
    let a = x[0];
    let b = x[1];
    a*a + b*b
}

fn main() {
    assert_eq!(norm2([3.0, 4.0]), 5.0*5.0);
}
