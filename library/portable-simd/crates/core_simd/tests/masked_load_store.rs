#![feature(portable_simd)]
use core_simd::simd::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn masked_load_store() {
    let mut arr = [u8::MAX; 7];

    u8x4::splat(0).store_select(&mut arr[5..], Mask::from_array([false, true, false, true]));
    // write to index 8 is OOB and dropped
    assert_eq!(arr, [255u8, 255, 255, 255, 255, 255, 0]);

    u8x4::from_array([0, 1, 2, 3]).store_select(&mut arr[1..], Mask::splat(true));
    assert_eq!(arr, [255u8, 0, 1, 2, 3, 255, 0]);

    // read from index 8 is OOB and dropped
    assert_eq!(
        u8x4::load_or(&arr[4..], u8x4::splat(42)),
        u8x4::from_array([3, 255, 0, 42])
    );
    assert_eq!(
        u8x4::load_select(
            &arr[4..],
            Mask::from_array([true, false, true, true]),
            u8x4::splat(42)
        ),
        u8x4::from_array([3, 42, 0, 42])
    );
}
