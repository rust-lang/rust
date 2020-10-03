macro_rules! mask_tests {
    { $vector:ident, $scalar:ident } => {
        #[cfg(test)]
        mod $vector {
            use super::*;
            use helpers::lanewise::*;

            #[cfg(target_arch = "wasm32")]
            use wasm_bindgen_test::*;

            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_test_configure!(run_in_browser);

            fn from_slice(slice: &[bool]) -> core_simd::$vector {
                let mut value = core_simd::$vector::default();
                let value_slice: &mut [_] = value.as_mut();
                for (m, b) in value_slice.iter_mut().zip(slice.iter()) {
                    *m = (*b).into();
                }
                value
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

            const SET_SCALAR: core_simd::$scalar = core_simd::$scalar::new(true);
            const UNSET_SCALAR: core_simd::$scalar = core_simd::$scalar::new(false);
            const SET_VECTOR: core_simd::$vector = core_simd::$vector::splat(SET_SCALAR);
            const UNSET_VECTOR: core_simd::$vector = core_simd::$vector::splat(UNSET_SCALAR);

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitand() {
                let a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::BitAnd::bitand);
                assert_biteq!(a & b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitand_assign() {
                let mut a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::BitAnd::bitand);
                a &= b;
                assert_biteq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitand_scalar_rhs() {
                let a = from_slice(&A);
                let expected = a;
                assert_biteq!(a & SET_SCALAR, expected);
                assert_biteq!(a & UNSET_SCALAR, UNSET_VECTOR);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitand_scalar_lhs() {
                let a = from_slice(&A);
                let expected = a;
                assert_biteq!(SET_SCALAR & a, expected);
                assert_biteq!(UNSET_SCALAR & a, UNSET_VECTOR);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitand_assign_scalar() {
                let mut a = from_slice(&A);
                let expected = a;
                a &= SET_SCALAR;
                assert_biteq!(a, expected);
                a &= UNSET_SCALAR;
                assert_biteq!(a, UNSET_VECTOR);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitor() {
                let a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::BitOr::bitor);
                assert_biteq!(a | b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitor_assign() {
                let mut a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::BitOr::bitor);
                a |= b;
                assert_biteq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitor_scalar_rhs() {
                let a = from_slice(&A);
                assert_biteq!(a | UNSET_SCALAR, a);
                assert_biteq!(a | SET_SCALAR, SET_VECTOR);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitor_scalar_lhs() {
                let a = from_slice(&A);
                assert_biteq!(UNSET_SCALAR | a, a);
                assert_biteq!(SET_SCALAR | a, SET_VECTOR);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitor_assign_scalar() {
                let mut a = from_slice(&A);
                let expected = a;
                a |= UNSET_SCALAR;
                assert_biteq!(a, expected);
                a |= SET_SCALAR;
                assert_biteq!(a, SET_VECTOR);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitxor() {
                let a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::BitXor::bitxor);
                assert_biteq!(a ^ b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitxor_assign() {
                let mut a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::BitXor::bitxor);
                a ^= b;
                assert_biteq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitxor_scalar_rhs() {
                let a = from_slice(&A);
                let expected = apply_binary_scalar_rhs_lanewise(a, SET_SCALAR, core::ops::BitXor::bitxor);
                assert_biteq!(a ^ UNSET_SCALAR, a);
                assert_biteq!(a ^ SET_SCALAR, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitxor_scalar_lhs() {
                let a = from_slice(&A);
                let expected = apply_binary_scalar_lhs_lanewise(SET_SCALAR, a, core::ops::BitXor::bitxor);
                assert_biteq!(UNSET_SCALAR ^ a, a);
                assert_biteq!(SET_SCALAR ^ a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn bitxor_assign_scalar() {
                let mut a = from_slice(&A);
                let expected_unset = a;
                let expected_set = apply_binary_scalar_rhs_lanewise(a, SET_SCALAR, core::ops::BitXor::bitxor);
                a ^= UNSET_SCALAR;
                assert_biteq!(a, expected_unset);
                a ^= SET_SCALAR;
                assert_biteq!(a, expected_set);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn not() {
                let v = from_slice(&A);
                let expected = apply_unary_lanewise(v, core::ops::Not::not);
                assert_biteq!(!v, expected);
            }
        }
    }
}
