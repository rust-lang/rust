//! Traits related to testing.
//!
//! There are three main traits in this module:
//!
//! - `GenerateInput`: implemented on any types that create test cases.
//! - `TupleCall`: implemented on tuples to allow calling them as function arguments.
//! - `CheckOutput`: implemented on anything that is an output type for validation against an
//!   expected value.

use std::fmt;

use anyhow::{Context, bail, ensure};

use crate::{BaseName, Float, Int, MaybeOverride, Name, SpecialCase, TestResult};

/// Context passed to [`CheckOutput`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckCtx {
    /// Allowed ULP deviation
    pub ulp: u32,
    pub fn_name: Name,
    pub base_name: BaseName,
    /// Function name.
    pub fn_name_str: &'static str,
    /// Return the unsuffixed version of the function name.
    pub base_name_str: &'static str,
    /// Source of truth for tests.
    pub basis: CheckBasis,
}

impl CheckCtx {
    pub fn new(ulp: u32, fn_name: Name, basis: CheckBasis) -> Self {
        Self {
            ulp,
            fn_name,
            fn_name_str: fn_name.as_str(),
            base_name: fn_name.base_name(),
            base_name_str: fn_name.base_name().as_str(),
            basis,
        }
    }
}

/// Possible items to test against
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckBasis {
    /// Check against Musl's math sources.
    Musl,
    /// Check against infinite precision (MPFR).
    Mpfr,
}

/// Implement this on types that can generate a sequence of tuples for test input.
pub trait GenerateInput<TupleArgs> {
    fn get_cases(&self) -> impl Iterator<Item = TupleArgs>;
}

/// Trait for calling a function with a tuple as arguments.
///
/// Implemented on the tuple with the function signature as the generic (so we can use the same
/// tuple for multiple signatures).
pub trait TupleCall<Func>: fmt::Debug {
    type Output;
    fn call(self, f: Func) -> Self::Output;
}

/// A trait to implement on any output type so we can verify it in a generic way.
pub trait CheckOutput<Input>: Sized {
    /// Validate `self` (actual) and `expected` are the same.
    ///
    /// `input` is only used here for error messages.
    fn validate(self, expected: Self, input: Input, ctx: &CheckCtx) -> TestResult;
}

/// A helper trait to print something as hex with the correct number of nibbles, e.g. a `u32`
/// will always print with `0x` followed by 8 digits.
///
/// This is only used for printing errors so allocating is okay.
pub trait Hex: Copy {
    fn hex(self) -> String;
}

/* implement `TupleCall` */

impl<T1, R> TupleCall<fn(T1) -> R> for (T1,)
where
    T1: fmt::Debug,
{
    type Output = R;

    fn call(self, f: fn(T1) -> R) -> Self::Output {
        f(self.0)
    }
}

impl<T1, T2, R> TupleCall<fn(T1, T2) -> R> for (T1, T2)
where
    T1: fmt::Debug,
    T2: fmt::Debug,
{
    type Output = R;

    fn call(self, f: fn(T1, T2) -> R) -> Self::Output {
        f(self.0, self.1)
    }
}

impl<T1, T2, R> TupleCall<fn(T1, &mut T2) -> R> for (T1,)
where
    T1: fmt::Debug,
    T2: fmt::Debug + Default,
{
    type Output = (R, T2);

    fn call(self, f: fn(T1, &mut T2) -> R) -> Self::Output {
        let mut t2 = T2::default();
        (f(self.0, &mut t2), t2)
    }
}

impl<T1, T2, T3, R> TupleCall<fn(T1, T2, T3) -> R> for (T1, T2, T3)
where
    T1: fmt::Debug,
    T2: fmt::Debug,
    T3: fmt::Debug,
{
    type Output = R;

    fn call(self, f: fn(T1, T2, T3) -> R) -> Self::Output {
        f(self.0, self.1, self.2)
    }
}

impl<T1, T2, T3, R> TupleCall<fn(T1, T2, &mut T3) -> R> for (T1, T2)
where
    T1: fmt::Debug,
    T2: fmt::Debug,
    T3: fmt::Debug + Default,
{
    type Output = (R, T3);

    fn call(self, f: fn(T1, T2, &mut T3) -> R) -> Self::Output {
        let mut t3 = T3::default();
        (f(self.0, self.1, &mut t3), t3)
    }
}

impl<T1, T2, T3> TupleCall<for<'a> fn(T1, &'a mut T2, &'a mut T3)> for (T1,)
where
    T1: fmt::Debug,
    T2: fmt::Debug + Default,
    T3: fmt::Debug + Default,
{
    type Output = (T2, T3);

    fn call(self, f: for<'a> fn(T1, &'a mut T2, &'a mut T3)) -> Self::Output {
        let mut t2 = T2::default();
        let mut t3 = T3::default();
        f(self.0, &mut t2, &mut t3);
        (t2, t3)
    }
}

/* implement `Hex` */

impl<T1> Hex for (T1,)
where
    T1: Hex,
{
    fn hex(self) -> String {
        format!("({},)", self.0.hex())
    }
}

impl<T1, T2> Hex for (T1, T2)
where
    T1: Hex,
    T2: Hex,
{
    fn hex(self) -> String {
        format!("({}, {})", self.0.hex(), self.1.hex())
    }
}

impl<T1, T2, T3> Hex for (T1, T2, T3)
where
    T1: Hex,
    T2: Hex,
    T3: Hex,
{
    fn hex(self) -> String {
        format!("({}, {}, {})", self.0.hex(), self.1.hex(), self.2.hex())
    }
}

/* trait implementations for ints */

macro_rules! impl_int {
    ($($ty:ty),*) => {
        $(
            impl Hex for $ty {
                fn hex(self) -> String {
                    format!("{self:#0width$x}", width = ((Self::BITS / 4) + 2) as usize)
                }
            }

            impl<Input> $crate::CheckOutput<Input> for $ty
            where
                Input: Hex + fmt::Debug,
                SpecialCase: MaybeOverride<Input>,
            {
                fn validate<'a>(
                    self,
                    expected: Self,
                    input: Input,
                    ctx: &$crate::CheckCtx,
                ) -> TestResult {
                    validate_int(self, expected, input, ctx)
                }
            }
        )*
    };
}

fn validate_int<I, Input>(actual: I, expected: I, input: Input, ctx: &CheckCtx) -> TestResult
where
    I: Int + Hex,
    Input: Hex + fmt::Debug,
    SpecialCase: MaybeOverride<Input>,
{
    if let Some(res) = SpecialCase::check_int(input, actual, expected, ctx) {
        return res;
    }

    anyhow::ensure!(
        actual == expected,
        "\
        \n    input:    {input:?} {ibits}\
        \n    expected: {expected:<22?} {expbits}\
        \n    actual:   {actual:<22?} {actbits}\
        ",
        actbits = actual.hex(),
        expbits = expected.hex(),
        ibits = input.hex(),
    );

    Ok(())
}

impl_int!(u32, i32, u64, i64);

/* trait implementations for floats */

macro_rules! impl_float {
    ($($ty:ty),*) => {
        $(
            impl Hex for $ty {
                fn hex(self) -> String {
                    format!(
                        "{:#0width$x}",
                        self.to_bits(),
                        width = ((Self::BITS / 4) + 2) as usize
                    )
                }
            }

            impl<Input> $crate::CheckOutput<Input> for $ty
            where
                Input: Hex + fmt::Debug,
                SpecialCase: MaybeOverride<Input>,
            {
                fn validate<'a>(
                    self,
                    expected: Self,
                    input: Input,
                    ctx: &$crate::CheckCtx,
                ) -> TestResult {
                    validate_float(self, expected, input, ctx)
                }
            }
        )*
    };
}

fn validate_float<F, Input>(actual: F, expected: F, input: Input, ctx: &CheckCtx) -> TestResult
where
    F: Float + Hex,
    Input: Hex + fmt::Debug,
    u32: TryFrom<F::SignedInt, Error: fmt::Debug>,
    SpecialCase: MaybeOverride<Input>,
{
    // Create a wrapper function so we only need to `.with_context` once.
    let inner = || -> TestResult {
        let mut allowed_ulp = ctx.ulp;

        // If the tested function requires a nonstandard test, run it here.
        if let Some(res) = SpecialCase::check_float(input, actual, expected, &mut allowed_ulp, ctx)
        {
            return res;
        }

        // Check when both are NaNs
        if actual.is_nan() && expected.is_nan() {
            // By default, NaNs have nothing special to check.
            return Ok(());
        } else if actual.is_nan() || expected.is_nan() {
            // Check when only one is a NaN
            bail!("real value != NaN")
        }

        // Make sure that the signs are the same before checing ULP to avoid wraparound
        let act_sig = actual.signum();
        let exp_sig = expected.signum();
        ensure!(act_sig == exp_sig, "mismatched signs {act_sig} {exp_sig}");

        if actual.is_infinite() ^ expected.is_infinite() {
            bail!("mismatched infinities");
        }

        let act_bits = actual.to_bits().signed();
        let exp_bits = expected.to_bits().signed();

        let ulp_diff = act_bits.checked_sub(exp_bits).unwrap().abs();

        let ulp_u32 = u32::try_from(ulp_diff)
            .map_err(|e| anyhow::anyhow!("{e:?}: ulp of {ulp_diff} exceeds u32::MAX"))?;

        ensure!(ulp_u32 <= allowed_ulp, "ulp {ulp_diff} > {allowed_ulp}",);

        Ok(())
    };

    inner().with_context(|| {
        format!(
            "\
            \n    input:    {input:?} {ibits}\
            \n    expected: {expected:<22?} {expbits}\
            \n    actual:   {actual:<22?} {actbits}\
            ",
            actbits = actual.hex(),
            expbits = expected.hex(),
            ibits = input.hex(),
        )
    })
}

impl_float!(f32, f64);

/* trait implementations for compound types */

/// Implement `CheckOutput` for combinations of types.
macro_rules! impl_tuples {
    ($(($a:ty, $b:ty);)*) => {
        $(
            impl<Input> CheckOutput<Input> for ($a, $b)
            where
                Input: Hex + fmt::Debug,
                SpecialCase: MaybeOverride<Input>,
              {
                fn validate<'a>(
                    self,
                    expected: Self,
                    input: Input,
                    ctx: &CheckCtx,
                ) -> TestResult {
                    self.0.validate(expected.0, input, ctx)
                        .and_then(|()| self.1.validate(expected.1, input, ctx))
                        .with_context(|| format!(
                            "full context:\
                            \n    input:    {input:?} {ibits}\
                            \n    expected: {expected:?} {expbits}\
                            \n    actual:   {self:?} {actbits}\
                            ",
                            actbits = self.hex(),
                            expbits = expected.hex(),
                            ibits = input.hex(),
                        ))
                }
            }
        )*
    };
}

impl_tuples!(
    (f32, i32);
    (f64, i32);
    (f32, f32);
    (f64, f64);
);
