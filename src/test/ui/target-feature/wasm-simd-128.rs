// only-wasm32
// check-pass

#![feature(wasm_target_feature)]

#[target_feature(enable = "wasm-simd128")]
unsafe fn foo() {}

fn main() {}
