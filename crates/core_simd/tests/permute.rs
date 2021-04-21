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

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn reverse() {
    let a = SimdU32::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(a.reverse().to_array(), [7, 6, 5, 4, 3, 2, 1, 0]);
}

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn interleave() {
    let a = SimdU32::from_array([0, 1, 2, 3, 4, 5, 6, 7]);
    let b = SimdU32::from_array([8, 9, 10, 11, 12, 13, 14, 15]);
    let (lo, hi) = a.interleave(b);
    assert_eq!(lo.to_array(), [0, 8, 1, 9, 2, 10, 3, 11]);
    assert_eq!(hi.to_array(), [4, 12, 5, 13, 6, 14, 7, 15]);
    let (even, odd) = lo.deinterleave(hi);
    assert_eq!(even, a);
    assert_eq!(odd, b);
}
