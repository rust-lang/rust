#![feature(portable_simd)]

macro_rules! unary_test {
    { $scalar:tt, $($func:tt),+ } => {
        test_helpers::test_lanes! {
            $(
            fn $func<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &core_simd::simd::Simd::<$scalar, LANES>::$func,
                    &$scalar::$func,
                    &|_| true,
                )
            }
            )*
        }
    }
}

macro_rules! binary_test {
    { $scalar:tt, $($func:tt),+ } => {
        test_helpers::test_lanes! {
            $(
            fn $func<const LANES: usize>() {
                test_helpers::test_binary_elementwise(
                    &core_simd::simd::Simd::<$scalar, LANES>::$func,
                    &$scalar::$func,
                    &|_, _| true,
                )
            }
            )*
        }
    }
}

macro_rules! ternary_test {
    { $scalar:tt, $($func:tt),+ } => {
        test_helpers::test_lanes! {
            $(
            fn $func<const LANES: usize>() {
                test_helpers::test_ternary_elementwise(
                    &core_simd::simd::Simd::<$scalar, LANES>::$func,
                    &$scalar::$func,
                    &|_, _, _| true,
                )
            }
            )*
        }
    }
}

macro_rules! impl_tests {
    { $scalar:tt } => {
        mod $scalar {
            use std_float::StdFloat;

            unary_test! { $scalar, sqrt, sin, cos, exp, exp2, ln, log2, log10, ceil, floor, round, trunc }
            binary_test! { $scalar, log }
            ternary_test! { $scalar, mul_add }

            test_helpers::test_lanes! {
                fn fract<const LANES: usize>() {
                    test_helpers::test_unary_elementwise_flush_subnormals(
                        &core_simd::simd::Simd::<$scalar, LANES>::fract,
                        &$scalar::fract,
                        &|_| true,
                    )
                }
            }
        }
    }
}

impl_tests! { f32 }
impl_tests! { f64 }
