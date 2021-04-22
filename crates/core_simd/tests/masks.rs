use core::convert::TryFrom;
use core_simd::{BitMask, Mask8, SimdI8, SimdMask8};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

#[test]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
fn mask_format_round_trip() {
    let ints = SimdI8::from_array([-1, 0, 0, -1]);

    let simd_mask = SimdMask8::try_from(ints).unwrap();

    let bitmask = BitMask::from(simd_mask);

    let opaque_mask = Mask8::from(bitmask);

    let simd_mask_returned = SimdMask8::from(opaque_mask);

    let ints_returned = SimdI8::from(simd_mask_returned);

    assert_eq!(ints_returned, ints);
}

macro_rules! test_mask_api {
    { $name:ident } => {
        #[allow(non_snake_case)]
        mod $name {
            #[cfg(target_arch = "wasm32")]
            use wasm_bindgen_test::*;

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn set_and_test() {
                let values = [true, false, false, true, false, false, true, false];
                let mut mask = core_simd::$name::<8>::splat(false);
                for (lane, value) in values.iter().copied().enumerate() {
                    mask.set(lane, value);
                }
                for (lane, value) in values.iter().copied().enumerate() {
                    assert_eq!(mask.test(lane), value);
                }
            }

            #[test]
            #[should_panic]
            fn set_invalid_lane() {
                let mut mask = core_simd::$name::<8>::splat(false);
                mask.set(8, true);
                let _ = mask;
            }

            #[test]
            #[should_panic]
            fn test_invalid_lane() {
                let mask = core_simd::$name::<8>::splat(false);
                let _ = mask.test(8);
            }

            #[test]
            fn any() {
                assert!(!core_simd::$name::<8>::splat(false).any());
                assert!(core_simd::$name::<8>::splat(true).any());
                let mut v = core_simd::$name::<8>::splat(false);
                v.set(2, true);
                assert!(v.any());
            }

            #[test]
            fn all() {
                assert!(!core_simd::$name::<8>::splat(false).all());
                assert!(core_simd::$name::<8>::splat(true).all());
                let mut v = core_simd::$name::<8>::splat(false);
                v.set(2, true);
                assert!(!v.all());
            }
        }
    }
}

mod mask_api {
    test_mask_api! { Mask8 }
    test_mask_api! { SimdMask8 }
    test_mask_api! { BitMask }
}
