pub mod array;

#[macro_use]
pub mod biteq;

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
impl_num! { i128 }
impl_num! { isize }
impl_num! { u8 }
impl_num! { u16 }
impl_num! { u32 }
impl_num! { u64 }
impl_num! { u128 }
impl_num! { usize }
impl_num! { f32 }
impl_num! { f64 }

impl<T: core::fmt::Debug + DefaultStrategy, const LANES: usize> DefaultStrategy for [T; LANES] {
    type Strategy = crate::array::UniformArrayStrategy<T::Strategy, Self>;
    fn default_strategy() -> Self::Strategy {
        Self::Strategy::new(T::default_strategy())
    }
}

pub fn test_1<A: core::fmt::Debug + DefaultStrategy>(
    f: impl Fn(A) -> proptest::test_runner::TestCaseResult,
) {
    let mut runner = proptest::test_runner::TestRunner::default();
    runner.run(&A::default_strategy(), f).unwrap();
}

pub fn test_2<A: core::fmt::Debug + DefaultStrategy, B: core::fmt::Debug + DefaultStrategy>(
    f: impl Fn(A, B) -> proptest::test_runner::TestCaseResult,
) {
    let mut runner = proptest::test_runner::TestRunner::default();
    runner
        .run(&(A::default_strategy(), B::default_strategy()), |(a, b)| {
            f(a, b)
        })
        .unwrap();
}

pub fn test_unary_elementwise<Scalar, ScalarResult, Vector, VectorResult, const LANES: usize>(
    fv: impl Fn(Vector) -> VectorResult,
    fs: impl Fn(Scalar) -> ScalarResult,
) where
    Scalar: Copy + Default + core::fmt::Debug + DefaultStrategy,
    ScalarResult: Copy + Default + biteq::BitEq + core::fmt::Debug + DefaultStrategy,
    Vector: Into<[Scalar; LANES]> + From<[Scalar; LANES]> + Copy,
    VectorResult: Into<[ScalarResult; LANES]> + From<[ScalarResult; LANES]> + Copy,
{
    test_1(|x: [Scalar; LANES]| {
        let result_1: [ScalarResult; LANES] = fv(x.into()).into();
        let result_2: [ScalarResult; LANES] = {
            let mut result = [ScalarResult::default(); LANES];
            for (i, o) in x.iter().zip(result.iter_mut()) {
                *o = fs(*i);
            }
            result
        };
        crate::prop_assert_biteq!(result_1, result_2);
        Ok(())
    });
}

pub fn test_binary_elementwise<
    Scalar1,
    Scalar2,
    ScalarResult,
    Vector1,
    Vector2,
    VectorResult,
    const LANES: usize,
>(
    fv: impl Fn(Vector1, Vector2) -> VectorResult,
    fs: impl Fn(Scalar1, Scalar2) -> ScalarResult,
) where
    Scalar1: Copy + Default + core::fmt::Debug + DefaultStrategy,
    Scalar2: Copy + Default + core::fmt::Debug + DefaultStrategy,
    ScalarResult: Copy + Default + biteq::BitEq + core::fmt::Debug + DefaultStrategy,
    Vector1: Into<[Scalar1; LANES]> + From<[Scalar1; LANES]> + Copy,
    Vector2: Into<[Scalar2; LANES]> + From<[Scalar2; LANES]> + Copy,
    VectorResult: Into<[ScalarResult; LANES]> + From<[ScalarResult; LANES]> + Copy,
{
    test_2(|x: [Scalar1; LANES], y: [Scalar2; LANES]| {
        let result_1: [ScalarResult; LANES] = fv(x.into(), y.into()).into();
        let result_2: [ScalarResult; LANES] = {
            let mut result = [ScalarResult::default(); LANES];
            for ((i1, i2), o) in x.iter().zip(y.iter()).zip(result.iter_mut()) {
                *o = fs(*i1, *i2);
            }
            result
        };
        crate::prop_assert_biteq!(result_1, result_2);
        Ok(())
    });
}

pub fn test_binary_scalar_rhs_elementwise<
    Scalar1,
    Scalar2,
    ScalarResult,
    Vector,
    VectorResult,
    const LANES: usize,
>(
    fv: impl Fn(Vector, Scalar2) -> VectorResult,
    fs: impl Fn(Scalar1, Scalar2) -> ScalarResult,
) where
    Scalar1: Copy + Default + core::fmt::Debug + DefaultStrategy,
    Scalar2: Copy + Default + core::fmt::Debug + DefaultStrategy,
    ScalarResult: Copy + Default + biteq::BitEq + core::fmt::Debug + DefaultStrategy,
    Vector: Into<[Scalar1; LANES]> + From<[Scalar1; LANES]> + Copy,
    VectorResult: Into<[ScalarResult; LANES]> + From<[ScalarResult; LANES]> + Copy,
{
    test_2(|x: [Scalar1; LANES], y: Scalar2| {
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

pub fn test_binary_scalar_lhs_elementwise<
    Scalar1,
    Scalar2,
    ScalarResult,
    Vector,
    VectorResult,
    const LANES: usize,
>(
    fv: impl Fn(Scalar1, Vector) -> VectorResult,
    fs: impl Fn(Scalar1, Scalar2) -> ScalarResult,
) where
    Scalar1: Copy + Default + core::fmt::Debug + DefaultStrategy,
    Scalar2: Copy + Default + core::fmt::Debug + DefaultStrategy,
    ScalarResult: Copy + Default + biteq::BitEq + core::fmt::Debug + DefaultStrategy,
    Vector: Into<[Scalar2; LANES]> + From<[Scalar2; LANES]> + Copy,
    VectorResult: Into<[ScalarResult; LANES]> + From<[ScalarResult; LANES]> + Copy,
{
    test_2(|x: Scalar1, y: [Scalar2; LANES]| {
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

#[macro_export]
#[doc(hidden)]
macro_rules! test_lanes_impl {
    {
        fn $test:ident<const $lanes:ident: usize>() $body:tt

        $($name:ident => $lanes_lit:literal,)*
    } => {
        mod $test {
            use super::*;
            $(
                #[test]
                #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
                fn $name() {
                    const $lanes: usize = $lanes_lit;
                    $body
                }
            )*
        }
    }
}

#[macro_export]
macro_rules! test_lanes {
    {
        $(fn $test:ident<const $lanes:ident: usize>() $body:tt)*
    } => {
        $(
        $crate::test_lanes_impl! {
            fn $test<const $lanes: usize>() $body

            lanes_2 => 2,
            lanes_3 => 3,
            lanes_4 => 4,
            lanes_7 => 7,
            lanes_8 => 8,
            lanes_16 => 16,
            lanes_32 => 32,
            lanes_64 => 64,
            lanes_128 => 128,
            lanes_256 => 256,
        }
        )*
    }
}
