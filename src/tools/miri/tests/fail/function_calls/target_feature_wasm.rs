//@only-target: wasm # tests WASM-specific behavior
//@compile-flags: -C target-feature=-simd128

fn main() {
    // Calling functions with `#[target_feature]` is not unsound on WASM, see #84988.
    // But if the compiler actually uses the target feature, it will lead to an error when the module is loaded.
    // We emulate this with an "unsupported" error.
    assert!(!cfg!(target_feature = "simd128"));
    simd128_fn(); //~ERROR: unavailable target features
}

#[target_feature(enable = "simd128")]
fn simd128_fn() {}
