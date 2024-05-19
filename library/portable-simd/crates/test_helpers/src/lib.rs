#![feature(powerpc_target_feature)]
#![cfg_attr(
    any(target_arch = "powerpc", target_arch = "powerpc64"),
    feature(stdarch_powerpc)
)]

pub mod array;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[macro_use]
pub mod biteq;

pub mod subnormals;
use subnormals::FlushSubnormals;

/// Specifies the default strategy for testing a type.
///
/// This strategy should be what "makes sense" to test.
pub trait DefaultStrategy {
    type Strategy: proptest::strategy::Strategy<Value = Self>;
    fn default_strategy() -> Self::Strategy;
}

macro_rules! impl_num {
    { $type:tt } => {
        impl DefaultStrategy for $type {
            type Strategy = proptest::num::$type::Any;
            fn default_strategy() -> Self::Strategy {
                proptest::num::$type::ANY
            }
        }
    }
}

impl_num! { i8 }
impl_num! { i16 }
impl_num! { i32 }
impl_num! { i64 }
impl_num! { isize }
impl_num! { u8 }
impl_num! { u16 }
impl_num! { u32 }
impl_num! { u64 }
impl_num! { usize }
impl_num! { f32 }
impl_num! { f64 }

impl<T> DefaultStrategy for *const T {
    type Strategy = proptest::strategy::Map<proptest::num::isize::Any, fn(isize) -> *const T>;
    fn default_strategy() -> Self::Strategy {
        fn map<T>(x: isize) -> *const T {
            x as _
        }
        use proptest::strategy::Strategy;
        proptest::num::isize::ANY.prop_map(map)
    }
}

impl<T> DefaultStrategy for *mut T {
    type Strategy = proptest::strategy::Map<proptest::num::isize::Any, fn(isize) -> *mut T>;
    fn default_strategy() -> Self::Strategy {
        fn map<T>(x: isize) -> *mut T {
            x as _
        }
        use proptest::strategy::Strategy;
        proptest::num::isize::ANY.prop_map(map)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl DefaultStrategy for u128 {
    type Strategy = proptest::num::u128::Any;
    fn default_strategy() -> Self::Strategy {
        proptest::num::u128::ANY
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl DefaultStrategy for i128 {
    type Strategy = proptest::num::i128::Any;
    fn default_strategy() -> Self::Strategy {
        proptest::num::i128::ANY
    }
}

#[cfg(target_arch = "wasm32")]
impl DefaultStrategy for u128 {
    type Strategy = crate::wasm::u128::Any;
    fn default_strategy() -> Self::Strategy {
        crate::wasm::u128::ANY
    }
}

#[cfg(target_arch = "wasm32")]
impl DefaultStrategy for i128 {
    type Strategy = crate::wasm::i128::Any;
    fn default_strategy() -> Self::Strategy {
        crate::wasm::i128::ANY
    }
}

impl<T: core::fmt::Debug + DefaultStrategy, const LANES: usize> DefaultStrategy for [T; LANES] {
    type Strategy = crate::array::UniformArrayStrategy<T::Strategy, Self>;
    fn default_strategy() -> Self::Strategy {
        Self::Strategy::new(T::default_strategy())
    }
}

#[cfg(not(miri))]
pub fn make_runner() -> proptest::test_runner::TestRunner {
    Default::default()
}
#[cfg(miri)]
pub fn make_runner() -> proptest::test_runner::TestRunner {
    // Only run a few tests on Miri
    proptest::test_runner::TestRunner::new(proptest::test_runner::Config::with_cases(4))
}

/// Test a function that takes a single value.
pub fn test_1<A: core::fmt::Debug + DefaultStrategy>(
    f: &dyn Fn(A) -> proptest::test_runner::TestCaseResult,
) {
    let mut runner = make_runner();
    runner.run(&A::default_strategy(), f).unwrap();
}

/// Test a function that takes two values.
pub fn test_2<A: core::fmt::Debug + DefaultStrategy, B: core::fmt::Debug + DefaultStrategy>(
    f: &dyn Fn(A, B) -> proptest::test_runner::TestCaseResult,
) {
    let mut runner = make_runner();
    runner
        .run(&(A::default_strategy(), B::default_strategy()), |(a, b)| {
            f(a, b)
        })
        .unwrap();
}

/// Test a function that takes two values.
pub fn test_3<
    A: core::fmt::Debug + DefaultStrategy,
    B: core::fmt::Debug + DefaultStrategy,
    C: core::fmt::Debug + DefaultStrategy,
>(
    f: &dyn Fn(A, B, C) -> proptest::test_runner::TestCaseResult,
) {
    let mut runner = make_runner();
    runner
        .run(
            &(
                A::default_strategy(),
                B::default_strategy(),
                C::default_strategy(),
            ),
            |(a, b, c)| f(a, b, c),
        )
        .unwrap();
}

/// Test a unary vector function against a unary scalar function, applied elementwise.
pub fn test_unary_elementwise<Scalar, ScalarResult, Vector, VectorResult, const LANES: usize>(
    fv: &dyn Fn(Vector) -> VectorResult,
    fs: &dyn Fn(Scalar) -> ScalarResult,
    check: &dyn Fn([Scalar; LANES]) -> bool,
) where
    Scalar: Copy + core::fmt::Debug + DefaultStrategy,
    ScalarResult: Copy + biteq::BitEq + core::fmt::Debug + DefaultStrategy,
    Vector: Into<[Scalar; LANES]> + From<[Scalar; LANES]> + Copy,
    VectorResult: Into<[ScalarResult; LANES]> + From<[ScalarResult; LANES]> + Copy,
{
    test_1(&|x: [Scalar; LANES]| {
        proptest::prop_assume!(check(x));
        let result_1: [ScalarResult; LANES] = fv(x.into()).into();
        let result_2: [ScalarResult; LANES] = x
            .iter()
            .copied()
            .map(fs)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        crate::prop_assert_biteq!(result_1, result_2);
        Ok(())
    });
}

/// Test a unary vector function against a unary scalar function, applied elementwise.
///
/// Where subnormals are flushed, use approximate equality.
pub fn test_unary_elementwise_flush_subnormals<
    Scalar,
    ScalarResult,
    Vector,
    VectorResult,
    const LANES: usize,
>(
    fv: &dyn Fn(Vector) -> VectorResult,
    fs: &dyn Fn(Scalar) -> ScalarResult,
    check: &dyn Fn([Scalar; LANES]) -> bool,
) where
    Scalar: Copy + core::fmt::Debug + DefaultStrategy + FlushSubnormals,
    ScalarResult: Copy + biteq::BitEq + core::fmt::Debug + DefaultStrategy + FlushSubnormals,
    Vector: Into<[Scalar; LANES]> + From<[Scalar; LANES]> + Copy,
    VectorResult: Into<[ScalarResult; LANES]> + From<[ScalarResult; LANES]> + Copy,
{
    let flush = |x: Scalar| subnormals::flush(fs(subnormals::flush_in(x)));
    test_1(&|x: [Scalar; LANES]| {
        proptest::prop_assume!(check(x));
        let result_v: [ScalarResult; LANES] = fv(x.into()).into();
        let result_s: [ScalarResult; LANES] = x
            .iter()
            .copied()
            .map(fs)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let result_sf: [ScalarResult; LANES] = x
            .iter()
            .copied()
            .map(flush)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        crate::prop_assert_biteq!(result_v, result_s, result_sf);
        Ok(())
    });
}

/// Test a unary vector function against a unary scalar function, applied elementwise.
#[inline(never)]
pub fn test_unary_mask_elementwise<Scalar, Vector, Mask, const LANES: usize>(
    fv: &dyn Fn(Vector) -> Mask,
    fs: &dyn Fn(Scalar) -> bool,
    check: &dyn Fn([Scalar; LANES]) -> bool,
) where
    Scalar: Copy + core::fmt::Debug + DefaultStrategy,
    Vector: Into<[Scalar; LANES]> + From<[Scalar; LANES]> + Copy,
    Mask: Into<[bool; LANES]> + From<[bool; LANES]> + Copy,
{
    test_1(&|x: [Scalar; LANES]| {
        proptest::prop_assume!(check(x));
        let result_1: [bool; LANES] = fv(x.into()).into();
        let result_2: [bool; LANES] = {
            let mut result = [false; LANES];
            for (i, o) in x.iter().zip(result.iter_mut()) {
                *o = fs(*i);
            }
            result
        };
        crate::prop_assert_biteq!(result_1, result_2);
        Ok(())
    });
}

/// Test a binary vector function against a binary scalar function, applied elementwise.
pub fn test_binary_elementwise<
    Scalar1,
    Scalar2,
    ScalarResult,
    Vector1,
    Vector2,
    VectorResult,
    const LANES: usize,
>(
    fv: &dyn Fn(Vector1, Vector2) -> VectorResult,
    fs: &dyn Fn(Scalar1, Scalar2) -> ScalarResult,
    check: &dyn Fn([Scalar1; LANES], [Scalar2; LANES]) -> bool,
) where
    Scalar1: Copy + core::fmt::Debug + DefaultStrategy,
    Scalar2: Copy + core::fmt::Debug + DefaultStrategy,
    ScalarResult: Copy + biteq::BitEq + core::fmt::Debug + DefaultStrategy,
    Vector1: Into<[Scalar1; LANES]> + From<[Scalar1; LANES]> + Copy,
    Vector2: Into<[Scalar2; LANES]> + From<[Scalar2; LANES]> + Copy,
    VectorResult: Into<[ScalarResult; LANES]> + From<[ScalarResult; LANES]> + Copy,
{
    test_2(&|x: [Scalar1; LANES], y: [Scalar2; LANES]| {
        proptest::prop_assume!(check(x, y));
        let result_1: [ScalarResult; LANES] = fv(x.into(), y.into()).into();
        let result_2: [ScalarResult; LANES] = x
            .iter()
            .copied()
            .zip(y.iter().copied())
            .map(|(x, y)| fs(x, y))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        crate::prop_assert_biteq!(result_1, result_2);
        Ok(())
    });
}

/// Test a binary vector function against a binary scalar function, applied elementwise.
///
/// Where subnormals are flushed, use approximate equality.
pub fn test_binary_elementwise_flush_subnormals<
    Scalar1,
    Scalar2,
    ScalarResult,
    Vector1,
    Vector2,
    VectorResult,
    const LANES: usize,
>(
    fv: &dyn Fn(Vector1, Vector2) -> VectorResult,
    fs: &dyn Fn(Scalar1, Scalar2) -> ScalarResult,
    check: &dyn Fn([Scalar1; LANES], [Scalar2; LANES]) -> bool,
) where
    Scalar1: Copy + core::fmt::Debug + DefaultStrategy + FlushSubnormals,
    Scalar2: Copy + core::fmt::Debug + DefaultStrategy + FlushSubnormals,
    ScalarResult: Copy + biteq::BitEq + core::fmt::Debug + DefaultStrategy + FlushSubnormals,
    Vector1: Into<[Scalar1; LANES]> + From<[Scalar1; LANES]> + Copy,
    Vector2: Into<[Scalar2; LANES]> + From<[Scalar2; LANES]> + Copy,
    VectorResult: Into<[ScalarResult; LANES]> + From<[ScalarResult; LANES]> + Copy,
{
    let flush = |x: Scalar1, y: Scalar2| {
        subnormals::flush(fs(subnormals::flush_in(x), subnormals::flush_in(y)))
    };
    test_2(&|x: [Scalar1; LANES], y: [Scalar2; LANES]| {
        proptest::prop_assume!(check(x, y));
        let result_v: [ScalarResult; LANES] = fv(x.into(), y.into()).into();
        let result_s: [ScalarResult; LANES] = x
            .iter()
            .copied()
            .zip(y.iter().copied())
            .map(|(x, y)| fs(x, y))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let result_sf: [ScalarResult; LANES] = x
            .iter()
            .copied()
            .zip(y.iter().copied())
            .map(|(x, y)| flush(x, y))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        crate::prop_assert_biteq!(result_v, result_s, result_sf);
        Ok(())
    });
}

/// Test a unary vector function against a unary scalar function, applied elementwise.
#[inline(never)]
pub fn test_binary_mask_elementwise<Scalar1, Scalar2, Vector1, Vector2, Mask, const LANES: usize>(
    fv: &dyn Fn(Vector1, Vector2) -> Mask,
    fs: &dyn Fn(Scalar1, Scalar2) -> bool,
    check: &dyn Fn([Scalar1; LANES], [Scalar2; LANES]) -> bool,
) where
    Scalar1: Copy + core::fmt::Debug + DefaultStrategy,
    Scalar2: Copy + core::fmt::Debug + DefaultStrategy,
    Vector1: Into<[Scalar1; LANES]> + From<[Scalar1; LANES]> + Copy,
    Vector2: Into<[Scalar2; LANES]> + From<[Scalar2; LANES]> + Copy,
    Mask: Into<[bool; LANES]> + From<[bool; LANES]> + Copy,
{
    test_2(&|x: [Scalar1; LANES], y: [Scalar2; LANES]| {
        proptest::prop_assume!(check(x, y));
        let result_v: [bool; LANES] = fv(x.into(), y.into()).into();
        let result_s: [bool; LANES] = x
            .iter()
            .copied()
            .zip(y.iter().copied())
            .map(|(x, y)| fs(x, y))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        crate::prop_assert_biteq!(result_v, result_s);
        Ok(())
    });
}

/// Test a binary vector-scalar function against a binary scalar function, applied elementwise.
#[inline(never)]
pub fn test_binary_scalar_rhs_elementwise<
    Scalar1,
    Scalar2,
    ScalarResult,
    Vector,
    VectorResult,
    const LANES: usize,
>(
    fv: &dyn Fn(Vector, Scalar2) -> VectorResult,
    fs: &dyn Fn(Scalar1, Scalar2) -> ScalarResult,
    check: &dyn Fn([Scalar1; LANES], Scalar2) -> bool,
) where
    Scalar1: Copy + Default + core::fmt::Debug + DefaultStrategy,
    Scalar2: Copy + Default + core::fmt::Debug + DefaultStrategy,
    ScalarResult: Copy + Default + biteq::BitEq + core::fmt::Debug + DefaultStrategy,
    Vector: Into<[Scalar1; LANES]> + From<[Scalar1; LANES]> + Copy,
    VectorResult: Into<[ScalarResult; LANES]> + From<[ScalarResult; LANES]> + Copy,
{
    test_2(&|x: [Scalar1; LANES], y: Scalar2| {
        proptest::prop_assume!(check(x, y));
        let result_1: [ScalarResult; LANES] = fv(x.into(), y).into();
        let result_2: [ScalarResult; LANES] = {
            let mut result = [ScalarResult::default(); LANES];
            for (i, o) in x.iter().zip(result.iter_mut()) {
                *o = fs(*i, y);
            }
            result
        };
        crate::prop_assert_biteq!(result_1, result_2);
        Ok(())
    });
}

/// Test a binary vector-scalar function against a binary scalar function, applied elementwise.
#[inline(never)]
pub fn test_binary_scalar_lhs_elementwise<
    Scalar1,
    Scalar2,
    ScalarResult,
    Vector,
    VectorResult,
    const LANES: usize,
>(
    fv: &dyn Fn(Scalar1, Vector) -> VectorResult,
    fs: &dyn Fn(Scalar1, Scalar2) -> ScalarResult,
    check: &dyn Fn(Scalar1, [Scalar2; LANES]) -> bool,
) where
    Scalar1: Copy + Default + core::fmt::Debug + DefaultStrategy,
    Scalar2: Copy + Default + core::fmt::Debug + DefaultStrategy,
    ScalarResult: Copy + Default + biteq::BitEq + core::fmt::Debug + DefaultStrategy,
    Vector: Into<[Scalar2; LANES]> + From<[Scalar2; LANES]> + Copy,
    VectorResult: Into<[ScalarResult; LANES]> + From<[ScalarResult; LANES]> + Copy,
{
    test_2(&|x: Scalar1, y: [Scalar2; LANES]| {
        proptest::prop_assume!(check(x, y));
        let result_1: [ScalarResult; LANES] = fv(x, y.into()).into();
        let result_2: [ScalarResult; LANES] = {
            let mut result = [ScalarResult::default(); LANES];
            for (i, o) in y.iter().zip(result.iter_mut()) {
                *o = fs(x, *i);
            }
            result
        };
        crate::prop_assert_biteq!(result_1, result_2);
        Ok(())
    });
}

/// Test a ternary vector function against a ternary scalar function, applied elementwise.
#[inline(never)]
pub fn test_ternary_elementwise<
    Scalar1,
    Scalar2,
    Scalar3,
    ScalarResult,
    Vector1,
    Vector2,
    Vector3,
    VectorResult,
    const LANES: usize,
>(
    fv: &dyn Fn(Vector1, Vector2, Vector3) -> VectorResult,
    fs: &dyn Fn(Scalar1, Scalar2, Scalar3) -> ScalarResult,
    check: &dyn Fn([Scalar1; LANES], [Scalar2; LANES], [Scalar3; LANES]) -> bool,
) where
    Scalar1: Copy + Default + core::fmt::Debug + DefaultStrategy,
    Scalar2: Copy + Default + core::fmt::Debug + DefaultStrategy,
    Scalar3: Copy + Default + core::fmt::Debug + DefaultStrategy,
    ScalarResult: Copy + Default + biteq::BitEq + core::fmt::Debug + DefaultStrategy,
    Vector1: Into<[Scalar1; LANES]> + From<[Scalar1; LANES]> + Copy,
    Vector2: Into<[Scalar2; LANES]> + From<[Scalar2; LANES]> + Copy,
    Vector3: Into<[Scalar3; LANES]> + From<[Scalar3; LANES]> + Copy,
    VectorResult: Into<[ScalarResult; LANES]> + From<[ScalarResult; LANES]> + Copy,
{
    test_3(
        &|x: [Scalar1; LANES], y: [Scalar2; LANES], z: [Scalar3; LANES]| {
            proptest::prop_assume!(check(x, y, z));
            let result_1: [ScalarResult; LANES] = fv(x.into(), y.into(), z.into()).into();
            let result_2: [ScalarResult; LANES] = {
                let mut result = [ScalarResult::default(); LANES];
                for ((i1, (i2, i3)), o) in
                    x.iter().zip(y.iter().zip(z.iter())).zip(result.iter_mut())
                {
                    *o = fs(*i1, *i2, *i3);
                }
                result
            };
            crate::prop_assert_biteq!(result_1, result_2);
            Ok(())
        },
    );
}

#[doc(hidden)]
#[macro_export]
macro_rules! test_lanes_helper {
    ($($(#[$meta:meta])* $fn_name:ident $lanes:literal;)+) => {
        $(
            #[test]
            $(#[$meta])*
            fn $fn_name() {
                implementation::<$lanes>();
            }
        )+
    };
    (
        $(#[$meta:meta])+;
        $($(#[$meta_before:meta])+ $fn_name_before:ident $lanes_before:literal;)*
        $fn_name:ident $lanes:literal;
        $($fn_name_rest:ident $lanes_rest:literal;)*
    ) => {
        $crate::test_lanes_helper!(
            $(#[$meta])+;
            $($(#[$meta_before])+ $fn_name_before $lanes_before;)*
            $(#[$meta])+ $fn_name $lanes;
            $($fn_name_rest $lanes_rest;)*
        );
    };
    (
        $(#[$meta_ignored:meta])+;
        $($(#[$meta:meta])+ $fn_name:ident $lanes:literal;)+
    ) => {
        $crate::test_lanes_helper!($($(#[$meta])+ $fn_name $lanes;)+);
    };
}

/// Expand a const-generic test into separate tests for each possible lane count.
#[macro_export]
macro_rules! test_lanes {
    {
        $(fn $test:ident<const $lanes:ident: usize>() $body:tt)*
    } => {
        $(
            mod $test {
                use super::*;

                fn implementation<const $lanes: usize>()
                where
                    core_simd::simd::LaneCount<$lanes>: core_simd::simd::SupportedLaneCount,
                $body

                #[cfg(target_arch = "wasm32")]
                wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

                $crate::test_lanes_helper!(
                    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)];
                    lanes_1 1;
                    lanes_2 2;
                    lanes_4 4;
                );

                #[cfg(not(miri))] // Miri intrinsic implementations are uniform and larger tests are sloooow
                $crate::test_lanes_helper!(
                    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)];
                    lanes_8 8;
                    lanes_16 16;
                    lanes_32 32;
                    lanes_64 64;
                );

                #[cfg(feature = "all_lane_counts")]
                $crate::test_lanes_helper!(
                    // test some odd and even non-power-of-2 lengths on miri
                    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)];
                    lanes_3 3;
                    lanes_5 5;
                    lanes_6 6;
                );

                #[cfg(feature = "all_lane_counts")]
                #[cfg(not(miri))] // Miri intrinsic implementations are uniform and larger tests are sloooow
                $crate::test_lanes_helper!(
                    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)];
                    lanes_7 7;
                    lanes_9 9;
                    lanes_10 10;
                    lanes_11 11;
                    lanes_12 12;
                    lanes_13 13;
                    lanes_14 14;
                    lanes_15 15;
                    lanes_17 17;
                    lanes_18 18;
                    lanes_19 19;
                    lanes_20 20;
                    lanes_21 21;
                    lanes_22 22;
                    lanes_23 23;
                    lanes_24 24;
                    lanes_25 25;
                    lanes_26 26;
                    lanes_27 27;
                    lanes_28 28;
                    lanes_29 29;
                    lanes_30 30;
                    lanes_31 31;
                    lanes_33 33;
                    lanes_34 34;
                    lanes_35 35;
                    lanes_36 36;
                    lanes_37 37;
                    lanes_38 38;
                    lanes_39 39;
                    lanes_40 40;
                    lanes_41 41;
                    lanes_42 42;
                    lanes_43 43;
                    lanes_44 44;
                    lanes_45 45;
                    lanes_46 46;
                    lanes_47 47;
                    lanes_48 48;
                    lanes_49 49;
                    lanes_50 50;
                    lanes_51 51;
                    lanes_52 52;
                    lanes_53 53;
                    lanes_54 54;
                    lanes_55 55;
                    lanes_56 56;
                    lanes_57 57;
                    lanes_58 58;
                    lanes_59 59;
                    lanes_60 60;
                    lanes_61 61;
                    lanes_62 62;
                    lanes_63 63;
                );
            }
        )*
    }
}

/// Expand a const-generic `#[should_panic]` test into separate tests for each possible lane count.
#[macro_export]
macro_rules! test_lanes_panic {
    {
        $(fn $test:ident<const $lanes:ident: usize>() $body:tt)*
    } => {
        $(
            mod $test {
                use super::*;

                fn implementation<const $lanes: usize>()
                where
                    core_simd::simd::LaneCount<$lanes>: core_simd::simd::SupportedLaneCount,
                $body

                $crate::test_lanes_helper!(
                    #[should_panic];
                    lanes_1 1;
                    lanes_2 2;
                    lanes_4 4;
                );

                #[cfg(not(miri))] // Miri intrinsic implementations are uniform and larger tests are sloooow
                $crate::test_lanes_helper!(
                    #[should_panic];
                    lanes_8 8;
                    lanes_16 16;
                    lanes_32 32;
                    lanes_64 64;
                );

                #[cfg(feature = "all_lane_counts")]
                $crate::test_lanes_helper!(
                    // test some odd and even non-power-of-2 lengths on miri
                    #[should_panic];
                    lanes_3 3;
                    lanes_5 5;
                    lanes_6 6;
                );

                #[cfg(feature = "all_lane_counts")]
                #[cfg(not(miri))] // Miri intrinsic implementations are uniform and larger tests are sloooow
                $crate::test_lanes_helper!(
                    #[should_panic];
                    lanes_7 7;
                    lanes_9 9;
                    lanes_10 10;
                    lanes_11 11;
                    lanes_12 12;
                    lanes_13 13;
                    lanes_14 14;
                    lanes_15 15;
                    lanes_17 17;
                    lanes_18 18;
                    lanes_19 19;
                    lanes_20 20;
                    lanes_21 21;
                    lanes_22 22;
                    lanes_23 23;
                    lanes_24 24;
                    lanes_25 25;
                    lanes_26 26;
                    lanes_27 27;
                    lanes_28 28;
                    lanes_29 29;
                    lanes_30 30;
                    lanes_31 31;
                    lanes_33 33;
                    lanes_34 34;
                    lanes_35 35;
                    lanes_36 36;
                    lanes_37 37;
                    lanes_38 38;
                    lanes_39 39;
                    lanes_40 40;
                    lanes_41 41;
                    lanes_42 42;
                    lanes_43 43;
                    lanes_44 44;
                    lanes_45 45;
                    lanes_46 46;
                    lanes_47 47;
                    lanes_48 48;
                    lanes_49 49;
                    lanes_50 50;
                    lanes_51 51;
                    lanes_52 52;
                    lanes_53 53;
                    lanes_54 54;
                    lanes_55 55;
                    lanes_56 56;
                    lanes_57 57;
                    lanes_58 58;
                    lanes_59 59;
                    lanes_60 60;
                    lanes_61 61;
                    lanes_62 62;
                    lanes_63 63;
                );
            }
        )*
    }
}
