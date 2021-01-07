#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

macro_rules! impl_unary_op_test {
    { $vector:ty, $scalar:ty, $trait:ident :: $fn:ident, $scalar_fn:expr } => {
        test_helpers::test_lanes! {
            fn $fn<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    <$vector as core::ops::$trait>::$fn,
                    $scalar_fn,
                    |_| true,
                );
            }
        }
    };
    { $vector:ty, $scalar:ty, $trait:ident :: $fn:ident } => {
        impl_unary_op_test! { $vector, $scalar, $trait::$fn, <$scalar as core::ops::$trait>::$fn }
    };
}

macro_rules! impl_binary_op_test {
    { $vector:ty, $scalar:ty, $trait:ident :: $fn:ident, $trait_assign:ident :: $fn_assign:ident, $scalar_fn:expr } => {
        mod $fn {
            use super::*;

            test_helpers::test_lanes! {
                fn normal<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        <$vector as core::ops::$trait>::$fn,
                        $scalar_fn,
                        |_, _| true,
                    );
                }

                fn scalar_rhs<const LANES: usize>() {
                    test_helpers::test_binary_scalar_rhs_elementwise(
                        <$vector as core::ops::$trait<$scalar>>::$fn,
                        $scalar_fn,
                        |_, _| true,
                    );
                }

                fn scalar_lhs<const LANES: usize>() {
                    test_helpers::test_binary_scalar_lhs_elementwise(
                        <$scalar as core::ops::$trait<$vector>>::$fn,
                        $scalar_fn,
                        |_, _| true,
                    );
                }

                fn assign<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        |mut a, b| { <$vector as core::ops::$trait_assign>::$fn_assign(&mut a, b); a },
                        $scalar_fn,
                        |_, _| true,
                    )
                }

                fn assign_scalar_rhs<const LANES: usize>() {
                    test_helpers::test_binary_scalar_rhs_elementwise(
                        |mut a, b| { <$vector as core::ops::$trait_assign<$scalar>>::$fn_assign(&mut a, b); a },
                        $scalar_fn,
                        |_, _| true,
                    )
                }
            }
        }
    };
    { $vector:ty, $scalar:ty, $trait:ident :: $fn:ident, $trait_assign:ident :: $fn_assign:ident } => {
        impl_binary_op_test! { $vector, $scalar, $trait::$fn, $trait_assign::$fn_assign, <$scalar as core::ops::$trait>::$fn }
    };
}

macro_rules! impl_binary_checked_op_test {
    { $vector:ty, $scalar:ty, $trait:ident :: $fn:ident, $trait_assign:ident :: $fn_assign:ident, $scalar_fn:expr, $check_fn:expr } => {
        mod $fn {
            use super::*;

            test_helpers::test_lanes! {
                fn normal<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        <$vector as core::ops::$trait>::$fn,
                        $scalar_fn,
                        |x, y| x.iter().zip(y.iter()).all(|(x, y)| $check_fn(*x, *y)),
                    );
                }

                fn scalar_rhs<const LANES: usize>() {
                    test_helpers::test_binary_scalar_rhs_elementwise(
                        <$vector as core::ops::$trait<$scalar>>::$fn,
                        $scalar_fn,
                        |x, y| x.iter().all(|x| $check_fn(*x, y)),
                    );
                }

                fn scalar_lhs<const LANES: usize>() {
                    test_helpers::test_binary_scalar_lhs_elementwise(
                        <$scalar as core::ops::$trait<$vector>>::$fn,
                        $scalar_fn,
                        |x, y| y.iter().all(|y| $check_fn(x, *y)),
                    );
                }

                fn assign<const LANES: usize>() {
                    test_helpers::test_binary_elementwise(
                        |mut a, b| { <$vector as core::ops::$trait_assign>::$fn_assign(&mut a, b); a },
                        $scalar_fn,
                        |x, y| x.iter().zip(y.iter()).all(|(x, y)| $check_fn(*x, *y)),
                    )
                }

                fn assign_scalar_rhs<const LANES: usize>() {
                    test_helpers::test_binary_scalar_rhs_elementwise(
                        |mut a, b| { <$vector as core::ops::$trait_assign<$scalar>>::$fn_assign(&mut a, b); a },
                        $scalar_fn,
                        |x, y| x.iter().all(|x| $check_fn(*x, y)),
                    )
                }
            }
        }
    };
    { $vector:ty, $scalar:ty, $trait:ident :: $fn:ident, $trait_assign:ident :: $fn_assign:ident, $check_fn:expr } => {
        impl_binary_nonzero_rhs_op_test! { $vector, $scalar, $trait::$fn, $trait_assign::$fn_assign, <$scalar as core::ops::$trait>::$fn, $check_fn }
    };
}

macro_rules! impl_signed_tests {
    { $vector:ident, $scalar:tt } => {
        mod $scalar {
            type Vector<const LANES: usize> = core_simd::$vector<LANES>;
            type Scalar = $scalar;

            test_helpers::test_lanes! {
                fn neg<const LANES: usize>() {
                    test_helpers::test_unary_elementwise(
                        <Vector<LANES> as core::ops::Neg>::neg,
                        <Scalar as core::ops::Neg>::neg,
                        |x| !x.contains(&Scalar::MIN),
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

impl_signed_tests! { SimdI8, i8 }
impl_signed_tests! { SimdI16, i16 }
impl_signed_tests! { SimdI32, i32 }
impl_signed_tests! { SimdI64, i64 }
impl_signed_tests! { SimdI128, i128 }
impl_signed_tests! { SimdIsize, isize }

impl_unsigned_tests! { SimdU8, u8 }
impl_unsigned_tests! { SimdU16, u16 }
impl_unsigned_tests! { SimdU32, u32 }
impl_unsigned_tests! { SimdU64, u64 }
impl_unsigned_tests! { SimdU128, u128 }
impl_unsigned_tests! { SimdUsize, usize }
