#![feature(portable_simd)]

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

use core_simd::simd::i32x4;

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn try_from_slice() {
    // Equal length
    assert_eq!(
        i32x4::try_from([1, 2, 3, 4].as_slice()).unwrap(),
        i32x4::from_array([1, 2, 3, 4])
    );

    // Slice length > vector length
    assert!(i32x4::try_from([1, 2, 3, 4, 5].as_slice()).is_err());

    // Slice length < vector length
    assert!(i32x4::try_from([1, 2, 3].as_slice()).is_err());
}
