//@only-target-wasm32: tests WASM-specific behavior
//@compile-flags: -C target-feature=-simd128

fn main() {
    // Calling functions with `#[target_feature]` is not unsound on WASM, see #84988
    assert!(!cfg!(target_feature = "simd128"));
    simd128_fn();
}

#[target_feature(enable = "simd128")]
fn simd128_fn() {}
