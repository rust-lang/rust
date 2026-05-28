macro_rules! mask_tests {
    { $vector:ident, $lanes:literal } => {
        #[cfg(test)]
        mod $vector {
            use core_simd::simd::$vector as Vector;
            const LANES: usize = $lanes;

            #[cfg(target_arch = "wasm32")]
            use wasm_bindgen_test::*;

            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_test_configure!(run_in_browser);

            fn from_slice(slice: &[bool]) -> Vector {
                let mut value = Vector::default();
                for (i, b) in slice.iter().take(LANES).enumerate() {
                    value.set(i, *b);
                }
                value
            }

            fn apply_unary_lanewise(x: Vector, f: impl Fn(bool) -> bool) -> Vector {
                let mut value = Vector::default();
                for i in 0..LANES {
                    value.set(i, f(x.test(i)));
                }
                value
            }

            fn apply_binary_lanewise(x: Vector, y: Vector, f: impl Fn(bool, bool) -> bool) -> Vector {
                let mut value = Vector::default();
                for i in 0..LANES {
                    value.set(i, f(x.test(i), y.test(i)));
                }
                value
            }

            fn apply_binary_scalar_lhs_lanewise(x: bool, mut y: Vector, f: impl Fn(bool, bool) -> bool) -> Vector {
                for i in 0..LANES {
                    y.set(i, f(x, y.test(i)));
                }
                y
            }

            fn apply_binary_scalar_rhs_lanewise(mut x: Vector, y: bool, f: impl Fn(bool, bool) -> bool) -> Vector {
                for i in 0..LANES {
                    x.set(i, f(x.test(i), y));
                }
                x
            }

            const A: [bool; 64] = [
                false, true, false, true, false, false, true, true,
                false, true, false, true, false, false, true, true,
                false, true, false, true, false, false, true, true,
                false, true, false, true, false, false, true, true,
                false, true, false, true, false, false, true, true,
                false, true, false, true, false, false, true, true,
                false, true, false, true, false, false, true, true,
                false, true, false, true, false, false, true, true,
            ];
            const B: [bool; 64] = [
                false, false, true, true, false, true, false, true,
                false, false, true, true, false, true, false, true,
                false, false, true, true, false, true, false, true,
                false, false, true, true, false, true, false, true,
                false, false, true, true, false, true, false, true,
                false, false, true, true, false, true, false, true,
                false, false, true, true, false, true, false, true,
                false, false, true, true, false, true, false, true,
            ];

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitand() {
                let a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::BitAnd::bitand);
                assert_eq!(a & b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitand_assign() {
                let mut a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::BitAnd::bitand);
                a &= b;
                assert_eq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitand_scalar_rhs() {
                let a = from_slice(&A);
                let expected = a;
                assert_eq!(a & true, expected);
                assert_eq!(a & false, Vector::splat(false));
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitand_scalar_lhs() {
                let a = from_slice(&A);
                let expected = a;
                assert_eq!(true & a, expected);
                assert_eq!(false & a, Vector::splat(false));
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitand_assign_scalar() {
                let mut a = from_slice(&A);
                let expected = a;
                a &= true;
                assert_eq!(a, expected);
                a &= false;
                assert_eq!(a, Vector::splat(false));
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitor() {
                let a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::BitOr::bitor);
                assert_eq!(a | b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitor_assign() {
                let mut a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::BitOr::bitor);
                a |= b;
                assert_eq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitor_scalar_rhs() {
                let a = from_slice(&A);
                assert_eq!(a | false, a);
                assert_eq!(a | true, Vector::splat(true));
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitor_scalar_lhs() {
                let a = from_slice(&A);
                assert_eq!(false | a, a);
                assert_eq!(true | a, Vector::splat(true));
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitor_assign_scalar() {
                let mut a = from_slice(&A);
                let expected = a;
                a |= false;
                assert_eq!(a, expected);
                a |= true;
                assert_eq!(a, Vector::splat(true));
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitxor() {
                let a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::BitXor::bitxor);
                assert_eq!(a ^ b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitxor_assign() {
                let mut a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::BitXor::bitxor);
                a ^= b;
                assert_eq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitxor_scalar_rhs() {
                let a = from_slice(&A);
                let expected = apply_binary_scalar_rhs_lanewise(a, true, core::ops::BitXor::bitxor);
                assert_eq!(a ^ false, a);
                assert_eq!(a ^ true, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitxor_scalar_lhs() {
                let a = from_slice(&A);
                let expected = apply_binary_scalar_lhs_lanewise(true, a, core::ops::BitXor::bitxor);
                assert_eq!(false ^ a, a);
                assert_eq!(true ^ a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitxor_assign_scalar() {
                let mut a = from_slice(&A);
                let expected_unset = a;
                let expected_set = apply_binary_scalar_rhs_lanewise(a, true, core::ops::BitXor::bitxor);
                a ^= false;
                assert_eq!(a, expected_unset);
                a ^= true;
                assert_eq!(a, expected_set);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn not() {
                let v = from_slice(&A);
                let expected = apply_unary_lanewise(v, core::ops::Not::not);
                assert_eq!(!v, expected);
            }
        }
    }
}
