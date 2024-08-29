//@ only-wasm32
//@ compile-flags:-C target-feature=-simd128
//@ build-pass

#![crate_type = "lib"]

#[cfg(target_feature = "simd128")]
compile_error!("simd128 target feature should be disabled");

// Calling functions with `#[target_feature]` is not unsound on WASM, see #84988.
// (It can still lead to a runtime error though so we'd be in our right to abort execution,
// just not to declare it UB.)
const A: () = simd128_fn();

#[target_feature(enable = "simd128")]
const fn simd128_fn() {}
