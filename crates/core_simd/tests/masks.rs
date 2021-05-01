#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::*;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

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

            #[test]
            fn roundtrip_int_conversion() {
                let values = [true, false, false, true, false, false, true, false];
                let mask = core_simd::$name::<8>::from_array(values);
                let int = mask.to_int();
                assert_eq!(int.to_array(), [-1, 0, 0, -1, 0, 0, -1, 0]);
                assert_eq!(core_simd::$name::<8>::from_int(int), mask);
            }

            #[test]
            fn to_bitmask() {
                let values = [
                    true, false, false, true, false, false, true, false,
                    true, true, false, false, false, false, false, true,
                ];
                let mask = core_simd::$name::<16>::from_array(values);
                assert_eq!(mask.to_bitmask(), [0b01001001, 0b10000011]);
            }
        }
    }
}

mod mask_api {
    test_mask_api! { Mask8 }
}
