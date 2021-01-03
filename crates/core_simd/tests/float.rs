#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

macro_rules! impl_op_test {
    { unary, $vector:ty, $scalar:ty, $trait:ident :: $fn:ident } => {
        test_helpers::test_lanes! {
            fn $fn<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    <$vector as core::ops::$trait>::$fn,
                    <$scalar as core::ops::$trait>::$fn,
                );
            }
        }
    };
    { binary, $vector:ty, $scalar:ty, $trait:ident :: $fn:ident, $trait_assign:ident :: $fn_assign:ident } => {
        mod $fn {
            use super::*;

            test_helpers::test_lanes! {
                fn normal<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        <$vector as core::ops::$trait>::$fn,
                        <$scalar as core::ops::$trait>::$fn,
                    );
                }

                fn scalar_rhs<const LANES: usize>() {
                    test_helpers::test_binary_scalar_rhs_elementwise(
                        <$vector as core::ops::$trait<$scalar>>::$fn,
                        <$scalar as core::ops::$trait>::$fn,
                    );
                }

                fn scalar_lhs<const LANES: usize>() {
                    test_helpers::test_binary_scalar_lhs_elementwise(
                        <$scalar as core::ops::$trait<$vector>>::$fn,
                        <$scalar as core::ops::$trait>::$fn,
                    );
                }

                fn assign<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        |mut a, b| { <$vector as core::ops::$trait_assign>::$fn_assign(&mut a, b); a },
                        |mut a, b| { <$scalar as core::ops::$trait_assign>::$fn_assign(&mut a, b); a },
                    )
                }

                fn assign_scalar_rhs<const LANES: usize>() {
                    test_helpers::test_binary_scalar_rhs_elementwise(
                        |mut a, b| { <$vector as core::ops::$trait_assign<$scalar>>::$fn_assign(&mut a, b); a },
                        |mut a, b| { <$scalar as core::ops::$trait_assign>::$fn_assign(&mut a, b); a },
                    )
                }
            }
        }
    };
}

macro_rules! impl_tests {
    { $vector:ident, $scalar:tt, $int_scalar:tt } => {
        mod $scalar {
            type Vector<const LANES: usize> = core_simd::$vector<LANES>;
            type Scalar = $scalar;
            type IntScalar = $int_scalar;
            
            impl_op_test! { unary, Vector<LANES>, Scalar, Neg::neg }
            impl_op_test! { binary, Vector<LANES>, Scalar, Add::add, AddAssign::add_assign }
            impl_op_test! { binary, Vector<LANES>, Scalar, Sub::sub, SubAssign::sub_assign }
            impl_op_test! { binary, Vector<LANES>, Scalar, Mul::mul, SubAssign::sub_assign }
            impl_op_test! { binary, Vector<LANES>, Scalar, Div::div, DivAssign::div_assign }
            impl_op_test! { binary, Vector<LANES>, Scalar, Rem::rem, RemAssign::rem_assign }

            test_helpers::test_lanes! {
                fn abs<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        Vector::<LANES>::abs,
                        Scalar::abs,
                    )
                }

                fn ceil<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        Vector::<LANES>::ceil,
                        Scalar::ceil,
                    )
                }

                fn floor<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        Vector::<LANES>::floor,
                        Scalar::floor,
                    )
                }

                fn round_from_int<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        Vector::<LANES>::round_from_int,
                        |x| x as Scalar,
                    )
                }

                fn to_int_unchecked<const LANES: usize>() {
                    // The maximum integer that can be represented by the equivalently sized float has
                    // all of the mantissa digits set to 1, pushed up to the MSB.
                    const ALL_MANTISSA_BITS: IntScalar = ((1 << <Scalar>::MANTISSA_DIGITS) - 1);
                    const MAX_REPRESENTABLE_VALUE: Scalar =
                        (ALL_MANTISSA_BITS << (core::mem::size_of::<Scalar>() * 8 - <Scalar>::MANTISSA_DIGITS as usize - 1)) as Scalar;

                    let mut runner = proptest::test_runner::TestRunner::default();
                    runner.run(
                        &test_helpers::array::UniformArrayStrategy::new(-MAX_REPRESENTABLE_VALUE..MAX_REPRESENTABLE_VALUE),
                        |x| {
                            let result_1 = unsafe { Vector::from_array(x).to_int_unchecked().to_array() };
                            let result_2 = {
                                let mut result = [0; LANES];
                                for (i, o) in x.iter().zip(result.iter_mut()) {
                                    *o = unsafe { i.to_int_unchecked() };
                                }
                                result
                            };
                            test_helpers::prop_assert_biteq!(result_1, result_2);
                            Ok(())
                        },
                    ).unwrap();
                }
            }
        }
    }
}

impl_tests! { SimdF32, f32, i32 }
impl_tests! { SimdF64, f64, i64 }
