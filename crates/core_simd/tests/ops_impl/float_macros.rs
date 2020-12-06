macro_rules! float_tests {
    { $vector:ident, $scalar:ident, $int_vector:ident, $int_scalar:ident } => {
        #[cfg(test)]
        mod $vector {
            use super::*;
            use helpers::lanewise::*;

            #[cfg(target_arch = "wasm32")]
            use wasm_bindgen_test::*;

            #[cfg(target_arch = "wasm32")]
            wasm_bindgen_test_configure!(run_in_browser);

            // TODO impl this as an associated fn on vectors
            fn from_slice(slice: &[$scalar]) -> core_simd::$vector {
                let mut value = core_simd::$vector::default();
                let value_slice: &mut [_] = value.as_mut();
                value_slice.copy_from_slice(&slice[0..value_slice.len()]);
                value
            }

            fn slice_chunks(slice: &[$scalar]) -> impl Iterator<Item = core_simd::$vector> + '_ {
                let lanes = core::mem::size_of::<core_simd::$vector>() / core::mem::size_of::<$scalar>();
                slice.chunks_exact(lanes).map(from_slice)
            }

            fn from_slice_int(slice: &[$int_scalar]) -> core_simd::$int_vector {
                let mut value = core_simd::$int_vector::default();
                let value_slice: &mut [_] = value.as_mut();
                value_slice.copy_from_slice(&slice[0..value_slice.len()]);
                value
            }

            fn slice_chunks_int(slice: &[$int_scalar]) -> impl Iterator<Item = core_simd::$int_vector> + '_ {
                let lanes = core::mem::size_of::<core_simd::$int_vector>() / core::mem::size_of::<$int_scalar>();
                slice.chunks_exact(lanes).map(from_slice_int)
            }

            const A: [$scalar; 16] = [0.,   1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.];
            const B: [$scalar; 16] = [16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31.];
            const C: [$scalar; 16] = [
                -0.0,
                0.0,
                -1.0,
                1.0,
                <$scalar>::MIN,
                <$scalar>::MAX,
                <$scalar>::INFINITY,
                <$scalar>::NEG_INFINITY,
                <$scalar>::MIN_POSITIVE,
                -<$scalar>::MIN_POSITIVE,
                <$scalar>::EPSILON,
                -<$scalar>::EPSILON,
                <$scalar>::NAN,
                -<$scalar>::NAN,
                // TODO: Would be nice to check sNaN...
                100.0 / 3.0,
                -100.0 / 3.0,
            ];

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn add() {
                let a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::Add::add);
                assert_biteq!(a + b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn add_assign() {
                let mut a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::Add::add);
                a += b;
                assert_biteq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn add_scalar_rhs() {
                let a = from_slice(&A);
                let b = 5.;
                let expected = apply_binary_scalar_rhs_lanewise(a, b, core::ops::Add::add);
                assert_biteq!(a + b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn add_scalar_lhs() {
                let a = 5.;
                let b = from_slice(&B);
                let expected = apply_binary_scalar_lhs_lanewise(a, b, core::ops::Add::add);
                assert_biteq!(a + b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn add_assign_scalar() {
                let mut a = from_slice(&A);
                let b = 5.;
                let expected = apply_binary_scalar_rhs_lanewise(a, b, core::ops::Add::add);
                a += b;
                assert_biteq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn sub() {
                let a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::Sub::sub);
                assert_biteq!(a - b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn sub_assign() {
                let mut a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::Sub::sub);
                a -= b;
                assert_biteq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn sub_scalar_rhs() {
                let a = from_slice(&A);
                let b = 5.;
                let expected = apply_binary_scalar_rhs_lanewise(a, b, core::ops::Sub::sub);
                assert_biteq!(a - b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn sub_scalar_lhs() {
                let a = 5.;
                let b = from_slice(&B);
                let expected = apply_binary_scalar_lhs_lanewise(a, b, core::ops::Sub::sub);
                assert_biteq!(a - b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn sub_assign_scalar() {
                let mut a = from_slice(&A);
                let b = 5.;
                let expected = apply_binary_scalar_rhs_lanewise(a, b, core::ops::Sub::sub);
                a -= b;
                assert_biteq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn mul() {
                let a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::Mul::mul);
                assert_biteq!(a * b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn mul_assign() {
                let mut a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::Mul::mul);
                a *= b;
                assert_biteq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn mul_scalar_rhs() {
                let a = from_slice(&A);
                let b = 5.;
                let expected = apply_binary_scalar_rhs_lanewise(a, b, core::ops::Mul::mul);
                assert_biteq!(a * b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn mul_scalar_lhs() {
                let a = 5.;
                let b = from_slice(&B);
                let expected = apply_binary_scalar_lhs_lanewise(a, b, core::ops::Mul::mul);
                assert_biteq!(a * b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn mul_assign_scalar() {
                let mut a = from_slice(&A);
                let b = 5.;
                let expected = apply_binary_scalar_rhs_lanewise(a, b, core::ops::Mul::mul);
                a *= b;
                assert_biteq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn div() {
                let a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::Div::div);
                assert_biteq!(a / b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn div_assign() {
                let mut a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::Div::div);
                a /= b;
                assert_biteq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn div_scalar_rhs() {
                let a = from_slice(&A);
                let b = 5.;
                let expected = apply_binary_scalar_rhs_lanewise(a, b, core::ops::Div::div);
                assert_biteq!(a / b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn div_scalar_lhs() {
                let a = 5.;
                let b = from_slice(&B);
                let expected = apply_binary_scalar_lhs_lanewise(a, b, core::ops::Div::div);
                assert_biteq!(a / b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn div_assign_scalar() {
                let mut a = from_slice(&A);
                let b = 5.;
                let expected = apply_binary_scalar_rhs_lanewise(a, b, core::ops::Div::div);
                a /= b;
                assert_biteq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn rem() {
                let a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::Rem::rem);
                assert_biteq!(a % b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn rem_assign() {
                let mut a = from_slice(&A);
                let b = from_slice(&B);
                let expected = apply_binary_lanewise(a, b, core::ops::Rem::rem);
                a %= b;
                assert_biteq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn rem_scalar_rhs() {
                let a = from_slice(&A);
                let b = 5.;
                let expected = apply_binary_scalar_rhs_lanewise(a, b, core::ops::Rem::rem);
                assert_biteq!(a % b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn rem_scalar_lhs() {
                let a = 5.;
                let b = from_slice(&B);
                let expected = apply_binary_scalar_lhs_lanewise(a, b, core::ops::Rem::rem);
                assert_biteq!(a % b, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn rem_assign_scalar() {
                let mut a = from_slice(&A);
                let b = 5.;
                let expected = apply_binary_scalar_rhs_lanewise(a, b, core::ops::Rem::rem);
                a %= b;
                assert_biteq!(a, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn neg() {
                let v = from_slice(&A);
                let expected = apply_unary_lanewise(v, core::ops::Neg::neg);
                assert_biteq!(-v, expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn neg_odd_floats() {
                for v in slice_chunks(&C) {
                    let expected = apply_unary_lanewise(v, core::ops::Neg::neg);
                    assert_biteq!(-v, expected);
                }
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn abs_negative() {
                let v = -from_slice(&A);
                let expected = apply_unary_lanewise(v, <$scalar>::abs);
                assert_biteq!(v.abs(), expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn abs_positive() {
                let v = from_slice(&B);
                let expected = apply_unary_lanewise(v, <$scalar>::abs);
                assert_biteq!(v.abs(), expected);
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn abs_odd_floats() {
                for v in slice_chunks(&C) {
                    let expected = apply_unary_lanewise(v, <$scalar>::abs);
                    assert_biteq!(v.abs(), expected);
                }
            }

            // TODO reenable after converting float ops to platform intrinsics
            /*
            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn ceil_odd_floats() {
                for v in slice_chunks(&C) {
                    let expected = apply_unary_lanewise(v, <$scalar>::ceil);
                    assert_biteq!(v.ceil(), expected);
                }
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn floor_odd_floats() {
                for v in slice_chunks(&C) {
                    let expected = apply_unary_lanewise(v, <$scalar>::floor);
                    assert_biteq!(v.floor(), expected);
                }
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn to_int_unchecked() {
                // The maximum integer that can be represented by the equivalently sized float has
                // all of the mantissa digits set to 1, pushed up to the MSB.
                const ALL_MANTISSA_BITS: $int_scalar = ((1 << <$scalar>::MANTISSA_DIGITS) - 1);
                const MAX_REPRESENTABLE_VALUE: $int_scalar =
                    ALL_MANTISSA_BITS << (core::mem::size_of::<$scalar>() * 8 - <$scalar>::MANTISSA_DIGITS as usize - 1);
                const VALUES: [$scalar; 16] = [
                    -0.0,
                    0.0,
                    -1.0,
                    1.0,
                    ALL_MANTISSA_BITS as $scalar,
                    -ALL_MANTISSA_BITS as $scalar,
                    MAX_REPRESENTABLE_VALUE as $scalar,
                    -MAX_REPRESENTABLE_VALUE as $scalar,
                    (MAX_REPRESENTABLE_VALUE / 2) as $scalar,
                    (-MAX_REPRESENTABLE_VALUE / 2) as $scalar,
                    <$scalar>::MIN_POSITIVE,
                    -<$scalar>::MIN_POSITIVE,
                    <$scalar>::EPSILON,
                    -<$scalar>::EPSILON,
                    100.0 / 3.0,
                    -100.0 / 3.0,
                ];

                for v in slice_chunks(&VALUES) {
                    let expected = apply_unary_lanewise(v, |x| unsafe { x.to_int_unchecked() });
                    assert_biteq!(unsafe { v.to_int_unchecked() }, expected);
                }
            }

            #[test]
            #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
            fn round_from_int() {
                const VALUES: [$int_scalar; 16] = [
                    0,
                    0,
                    1,
                    -1,
                    100,
                    -100,
                    200,
                    -200,
                    413,
                    -413,
                    1017,
                    -1017,
                    1234567,
                    -1234567,
                    <$int_scalar>::MAX,
                    <$int_scalar>::MIN,
                ];

                for v in slice_chunks_int(&VALUES) {
                    let expected = apply_unary_lanewise(v, |x| x as $scalar);
                    assert_biteq!(core_simd::$vector::round_from_int(v), expected);
                }
            }
            */
        }
    }
}
