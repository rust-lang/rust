// check-pass

#![feature(wasm_target_feature)]

#[cfg(any(target_arch = "wasm32", doc))]
#[target_feature(enable = "simd128")]
pub fn foo() {}
