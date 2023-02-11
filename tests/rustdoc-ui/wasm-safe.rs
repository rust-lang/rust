// check-pass

#[cfg(any(target_family = "wasm", doc))]
#[target_feature(enable = "simd128")]
pub fn foo() {}
