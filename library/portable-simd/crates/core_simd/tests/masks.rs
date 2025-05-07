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

            use core_simd::simd::Mask;

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn set_and_test() {
                let values = [true, false, false, true, false, false, true, false];
                let mut mask = Mask::<$type, 8>::splat(false);
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
                let mut mask = Mask::<$type, 8>::splat(false);
                mask.set(8, true);
                let _ = mask;
            }

            #[test]
            #[should_panic]
            fn test_invalid_lane() {
                let mask = Mask::<$type, 8>::splat(false);
                let _ = mask.test(8);
            }

            #[test]
            fn any() {
                assert!(!Mask::<$type, 8>::splat(false).any());
                assert!(Mask::<$type, 8>::splat(true).any());
                let mut v = Mask::<$type, 8>::splat(false);
                v.set(2, true);
                assert!(v.any());
            }

            #[test]
            fn all() {
                assert!(!Mask::<$type, 8>::splat(false).all());
                assert!(Mask::<$type, 8>::splat(true).all());
                let mut v = Mask::<$type, 8>::splat(false);
                v.set(2, true);
                assert!(!v.all());
            }

            #[test]
            fn roundtrip_int_conversion() {
                let values = [true, false, false, true, false, false, true, false];
                let mask = Mask::<$type, 8>::from_array(values);
                let int = mask.to_int();
                assert_eq!(int.to_array(), [-1, 0, 0, -1, 0, 0, -1, 0]);
                assert_eq!(Mask::<$type, 8>::from_int(int), mask);
            }

            #[test]
            fn roundtrip_bitmask_conversion() {
                let values = [
                    true, false, false, true, false, false, true, false,
                    true, true, false, false, false, false, false, true,
                ];
                let mask = Mask::<$type, 16>::from_array(values);
                let bitmask = mask.to_bitmask();
                assert_eq!(bitmask, 0b1000001101001001);
                assert_eq!(Mask::<$type, 16>::from_bitmask(bitmask), mask);
            }

            #[test]
            fn roundtrip_bitmask_conversion_short() {
                let values = [
                    false, false, false, true,
                ];
                let mask = Mask::<$type, 4>::from_array(values);
                let bitmask = mask.to_bitmask();
                assert_eq!(bitmask, 0b1000);
                assert_eq!(Mask::<$type, 4>::from_bitmask(bitmask), mask);

                let values = [true, false];
                let mask = Mask::<$type, 2>::from_array(values);
                let bitmask = mask.to_bitmask();
                assert_eq!(bitmask, 0b01);
                assert_eq!(Mask::<$type, 2>::from_bitmask(bitmask), mask);
            }

            #[test]
            fn roundtrip_bitmask_conversion_odd() {
                let values = [
                    true, false, true, false, true, true, false, false, false, true, true,
                ];
                let mask = Mask::<$type, 11>::from_array(values);
                let bitmask = mask.to_bitmask();
                assert_eq!(bitmask, 0b11000110101);
                assert_eq!(Mask::<$type, 11>::from_bitmask(bitmask), mask);
            }


            #[test]
            fn cast() {
                fn cast_impl<T: core_simd::simd::MaskElement>()
                where
                    Mask<$type, 8>: Into<Mask<T, 8>>,
                {
                    let values = [true, false, false, true, false, false, true, false];
                    let mask = Mask::<$type, 8>::from_array(values);

                    let cast_mask = mask.cast::<T>();
                    assert_eq!(values, cast_mask.to_array());

                    let into_mask: Mask<T, 8> = mask.into();
                    assert_eq!(values, into_mask.to_array());
                }

                cast_impl::<i8>();
                cast_impl::<i16>();
                cast_impl::<i32>();
                cast_impl::<i64>();
                cast_impl::<isize>();
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
    use core_simd::simd::Mask;
    let values = [true, false, false, true, false, false, true, false];
    assert_eq!(
        Mask::<i8, 8>::from_array(values),
        Mask::<i32, 8>::from_array(values).into()
    );
}
