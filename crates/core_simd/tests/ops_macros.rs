#[macro_export]
macro_rules! impl_unary_op_test {
    { $vector:ty, $scalar:ty, $trait:ident :: $fn:ident, $scalar_fn:expr } => {
        test_helpers::test_lanes! {
            fn $fn<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &<$vector as core::ops::$trait>::$fn,
                    &$scalar_fn,
                    &|_| true,
                );
            }
        }
    };
    { $vector:ty, $scalar:ty, $trait:ident :: $fn:ident } => {
        impl_unary_op_test! { $vector, $scalar, $trait::$fn, <$scalar as core::ops::$trait>::$fn }
    };
}

#[macro_export]
macro_rules! impl_binary_op_test {
    { $vector:ty, $scalar:ty, $trait:ident :: $fn:ident, $trait_assign:ident :: $fn_assign:ident, $scalar_fn:expr } => {
        mod $fn {
            use super::*;

            test_helpers::test_lanes! {
                fn normal<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        &<$vector as core::ops::$trait>::$fn,
                        &$scalar_fn,
                        &|_, _| true,
                    );
                }

                fn scalar_rhs<const LANES: usize>() {
                    test_helpers::test_binary_scalar_rhs_elementwise(
                        &<$vector as core::ops::$trait<$scalar>>::$fn,
                        &$scalar_fn,
                        &|_, _| true,
                    );
                }

                fn scalar_lhs<const LANES: usize>() {
                    test_helpers::test_binary_scalar_lhs_elementwise(
                        &<$scalar as core::ops::$trait<$vector>>::$fn,
                        &$scalar_fn,
                        &|_, _| true,
                    );
                }

                fn assign<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        &|mut a, b| { <$vector as core::ops::$trait_assign>::$fn_assign(&mut a, b); a },
                        &$scalar_fn,
                        &|_, _| true,
                    );
                }

                fn assign_scalar_rhs<const LANES: usize>() {
                    test_helpers::test_binary_scalar_rhs_elementwise(
                        &|mut a, b| { <$vector as core::ops::$trait_assign<$scalar>>::$fn_assign(&mut a, b); a },
                        &$scalar_fn,
                        &|_, _| true,
                    );
                }
            }
        }
    };
    { $vector:ty, $scalar:ty, $trait:ident :: $fn:ident, $trait_assign:ident :: $fn_assign:ident } => {
        impl_binary_op_test! { $vector, $scalar, $trait::$fn, $trait_assign::$fn_assign, <$scalar as core::ops::$trait>::$fn }
    };
}

#[macro_export]
macro_rules! impl_binary_checked_op_test {
    { $vector:ty, $scalar:ty, $trait:ident :: $fn:ident, $trait_assign:ident :: $fn_assign:ident, $scalar_fn:expr, $check_fn:expr } => {
        mod $fn {
            use super::*;

            test_helpers::test_lanes! {
                fn normal<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        &<$vector as core::ops::$trait>::$fn,
                        &$scalar_fn,
                        &|x, y| x.iter().zip(y.iter()).all(|(x, y)| $check_fn(*x, *y)),
                    );
                }

                fn scalar_rhs<const LANES: usize>() {
                    test_helpers::test_binary_scalar_rhs_elementwise(
                        &<$vector as core::ops::$trait<$scalar>>::$fn,
                        &$scalar_fn,
                        &|x, y| x.iter().all(|x| $check_fn(*x, y)),
                    );
                }

                fn scalar_lhs<const LANES: usize>() {
                    test_helpers::test_binary_scalar_lhs_elementwise(
                        &<$scalar as core::ops::$trait<$vector>>::$fn,
                        &$scalar_fn,
                        &|x, y| y.iter().all(|y| $check_fn(x, *y)),
                    );
                }

                fn assign<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        &|mut a, b| { <$vector as core::ops::$trait_assign>::$fn_assign(&mut a, b); a },
                        &$scalar_fn,
                        &|x, y| x.iter().zip(y.iter()).all(|(x, y)| $check_fn(*x, *y)),
                    )
                }

                fn assign_scalar_rhs<const LANES: usize>() {
                    test_helpers::test_binary_scalar_rhs_elementwise(
                        &|mut a, b| { <$vector as core::ops::$trait_assign<$scalar>>::$fn_assign(&mut a, b); a },
                        &$scalar_fn,
                        &|x, y| x.iter().all(|x| $check_fn(*x, y)),
                    )
                }
            }
        }
    };
    { $vector:ty, $scalar:ty, $trait:ident :: $fn:ident, $trait_assign:ident :: $fn_assign:ident, $check_fn:expr } => {
        impl_binary_nonzero_rhs_op_test! { $vector, $scalar, $trait::$fn, $trait_assign::$fn_assign, <$scalar as core::ops::$trait>::$fn, $check_fn }
    };
}

#[macro_export]
macro_rules! impl_signed_tests {
    { $vector:ident, $scalar:tt } => {
        mod $scalar {
            type Vector<const LANES: usize> = core_simd::$vector<LANES>;
            type Scalar = $scalar;

            test_helpers::test_lanes! {
                fn neg<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &<Vector<LANES> as core::ops::Neg>::neg,
                        &<Scalar as core::ops::Neg>::neg,
                        &|x| !x.contains(&Scalar::MIN),
                    );
                }
            }

            impl_binary_op_test!(Vector<LANES>, Scalar, Add::add, AddAssign::add_assign, Scalar::wrapping_add);
            impl_binary_op_test!(Vector<LANES>, Scalar, Sub::sub, SubAssign::sub_assign, Scalar::wrapping_sub);
            impl_binary_op_test!(Vector<LANES>, Scalar, Mul::mul, MulAssign::mul_assign, Scalar::wrapping_mul);
            impl_binary_checked_op_test!(Vector<LANES>, Scalar, Div::div, DivAssign::div_assign, Scalar::wrapping_div, |x, y| y != 0 && !(x == Scalar::MIN && y == -1));
            impl_binary_checked_op_test!(Vector<LANES>, Scalar, Rem::rem, RemAssign::rem_assign, Scalar::wrapping_rem, |x, y| y != 0 && !(x == Scalar::MIN && y == -1));

            impl_unary_op_test!(Vector<LANES>, Scalar, Not::not);
            impl_binary_op_test!(Vector<LANES>, Scalar, BitAnd::bitand, BitAndAssign::bitand_assign);
            impl_binary_op_test!(Vector<LANES>, Scalar, BitOr::bitor, BitOrAssign::bitor_assign);
            impl_binary_op_test!(Vector<LANES>, Scalar, BitXor::bitxor, BitXorAssign::bitxor_assign);
        }
    }
}

#[macro_export]
macro_rules! impl_unsigned_tests {
    { $vector:ident, $scalar:tt } => {
        mod $scalar {
            type Vector<const LANES: usize> = core_simd::$vector<LANES>;
            type Scalar = $scalar;

            impl_binary_op_test!(Vector<LANES>, Scalar, Add::add, AddAssign::add_assign, Scalar::wrapping_add);
            impl_binary_op_test!(Vector<LANES>, Scalar, Sub::sub, SubAssign::sub_assign, Scalar::wrapping_sub);
            impl_binary_op_test!(Vector<LANES>, Scalar, Mul::mul, MulAssign::mul_assign, Scalar::wrapping_mul);
            impl_binary_checked_op_test!(Vector<LANES>, Scalar, Div::div, DivAssign::div_assign, Scalar::wrapping_div, |_, y| y != 0);
            impl_binary_checked_op_test!(Vector<LANES>, Scalar, Rem::rem, RemAssign::rem_assign, Scalar::wrapping_rem, |_, y| y != 0);

            impl_unary_op_test!(Vector<LANES>, Scalar, Not::not);
            impl_binary_op_test!(Vector<LANES>, Scalar, BitAnd::bitand, BitAndAssign::bitand_assign);
            impl_binary_op_test!(Vector<LANES>, Scalar, BitOr::bitor, BitOrAssign::bitor_assign);
            impl_binary_op_test!(Vector<LANES>, Scalar, BitXor::bitxor, BitXorAssign::bitxor_assign);
        }
    }
}

#[macro_export]
macro_rules! impl_float_tests {
    { $vector:ident, $scalar:tt, $int_scalar:tt } => {
        mod $scalar {
            type Vector<const LANES: usize> = core_simd::$vector<LANES>;
            type Scalar = $scalar;
            type IntScalar = $int_scalar;

            impl_unary_op_test!(Vector<LANES>, Scalar, Neg::neg);
            impl_binary_op_test!(Vector<LANES>, Scalar, Add::add, AddAssign::add_assign);
            impl_binary_op_test!(Vector<LANES>, Scalar, Sub::sub, SubAssign::sub_assign);
            impl_binary_op_test!(Vector<LANES>, Scalar, Mul::mul, SubAssign::sub_assign);
            impl_binary_op_test!(Vector<LANES>, Scalar, Div::div, DivAssign::div_assign);
            impl_binary_op_test!(Vector<LANES>, Scalar, Rem::rem, RemAssign::rem_assign);

            test_helpers::test_lanes! {
                fn abs<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        &Vector::<LANES>::abs,
                        &Scalar::abs,
                        &|_| true,
                    )
                }

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

                fn round_from_int<const LANES: usize>() {
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
