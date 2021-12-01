// Test that we handle all our "auto-deref" cases correctly.
#![feature(portable_simd)]
use core_simd::f32x4;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn deref() {
    let x = f32x4::splat(1.0);
    let y = f32x4::splat(2.0);
    let a = &x;
    let b = &y;
    assert_eq!(f32x4::splat(3.0), x + y);
    assert_eq!(f32x4::splat(3.0), x + b);
    assert_eq!(f32x4::splat(3.0), a + y);
    assert_eq!(f32x4::splat(3.0), a + b);
}
