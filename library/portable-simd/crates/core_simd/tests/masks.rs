#![feature(portable_simd)]

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

macro_rules! test_mask_api {
    { $type:ident } => {
        #[allow(non_snake_case)]
        mod $type {
            #[cfg(target_arch = "wasm32")]
            use wasm_bindgen_test::*;

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn set_and_test() {
                let values = [true, false, false, true, false, false, true, false];
                let mut mask = core_simd::Mask::<$type, 8>::splat(false);
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
                let mut mask = core_simd::Mask::<$type, 8>::splat(false);
                mask.set(8, true);
                let _ = mask;
            }

            #[test]
            #[should_panic]
            fn test_invalid_lane() {
                let mask = core_simd::Mask::<$type, 8>::splat(false);
                let _ = mask.test(8);
            }

            #[test]
            fn any() {
                assert!(!core_simd::Mask::<$type, 8>::splat(false).any());
                assert!(core_simd::Mask::<$type, 8>::splat(true).any());
                let mut v = core_simd::Mask::<$type, 8>::splat(false);
                v.set(2, true);
                assert!(v.any());
            }

            #[test]
            fn all() {
                assert!(!core_simd::Mask::<$type, 8>::splat(false).all());
                assert!(core_simd::Mask::<$type, 8>::splat(true).all());
                let mut v = core_simd::Mask::<$type, 8>::splat(false);
                v.set(2, true);
                assert!(!v.all());
            }

            #[test]
            fn roundtrip_int_conversion() {
                let values = [true, false, false, true, false, false, true, false];
                let mask = core_simd::Mask::<$type, 8>::from_array(values);
                let int = mask.to_int();
                assert_eq!(int.to_array(), [-1, 0, 0, -1, 0, 0, -1, 0]);
                assert_eq!(core_simd::Mask::<$type, 8>::from_int(int), mask);
            }

            #[cfg(feature = "generic_const_exprs")]
            #[test]
            fn roundtrip_bitmask_conversion() {
                let values = [
                    true, false, false, true, false, false, true, false,
                    true, true, false, false, false, false, false, true,
                ];
                let mask = core_simd::Mask::<$type, 16>::from_array(values);
                let bitmask = mask.to_bitmask();
                assert_eq!(bitmask, [0b01001001, 0b10000011]);
                assert_eq!(core_simd::Mask::<$type, 16>::from_bitmask(bitmask), mask);
            }
        }
    }
}

mod mask_api {
    test_mask_api! { i8 }
    test_mask_api! { i16 }
    test_mask_api! { i32 }
    test_mask_api! { i64 }
    test_mask_api! { isize }
}

#[test]
fn convert() {
    let values = [true, false, false, true, false, false, true, false];
    assert_eq!(
        core_simd::Mask::<i8, 8>::from_array(values),
        core_simd::Mask::<i32, 8>::from_array(values).into()
    );
}
