#![feature(portable_simd)]

macro_rules! float_rounding_test {
    { $scalar:tt, $int_scalar:tt } => {
        mod $scalar {
            use std_float::StdFloat;

            type Vector<const LANES: usize> = core_simd::Simd<$scalar, LANES>;
            type Scalar = $scalar;
            type IntScalar = $int_scalar;

            #[cfg(feature = "std")]
            test_helpers::test_lanes! {
                fn ceil<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::ceil,
                        &Scalar::ceil,
                        &|_| true,
                    )
                }

                fn floor<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::floor,
                        &Scalar::floor,
                        &|_| true,
                    )
                }

                fn round<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::round,
                        &Scalar::round,
                        &|_| true,
                    )
                }

                fn trunc<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::trunc,
                        &Scalar::trunc,
                        &|_| true,
                    )
                }

                fn fract<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::fract,
                        &Scalar::fract,
                        &|_| true,
                    )
                }
            }

            test_helpers::test_lanes! {
                fn from_int<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::round_from_int,
                        &|x| x as Scalar,
                        &|_| true,
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

float_rounding_test! { f32, i32 }
float_rounding_test! { f64, i64 }
