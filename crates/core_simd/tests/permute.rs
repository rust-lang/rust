use core_simd::SimdU32;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn simple_shuffle() {
    let a = SimdU32::from_array([2, 4, 1, 9]);
    let b = a;
    assert_eq!(a.shuffle::<{ [3, 1, 4, 6] }>(b).to_array(), [9, 4, 2, 1]);
}
