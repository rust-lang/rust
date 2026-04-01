//@ check-pass

#[cfg(any(target_arch = "wasm32", doc))]
#[target_feature(enable = "simd128")]
pub fn foo() {}
