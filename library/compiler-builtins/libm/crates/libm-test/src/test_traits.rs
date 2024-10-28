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

use crate::{Float, Hex, Int, MaybeOverride, SpecialCase, TestResult};

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

/// Context passed to [`CheckOutput`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckCtx {
    /// Allowed ULP deviation
    pub ulp: u32,
    /// Function name.
    pub fname: &'static str,
    /// Return the unsuffixed version of the function name.
    pub canonical_name: &'static str,
    /// Source of truth for tests.
    pub basis: CheckBasis,
}

impl CheckCtx {
    pub fn new(ulp: u32, fname: &'static str, basis: CheckBasis) -> Self {
        let canonical_fname = crate::canonical_name(fname);
        Self { ulp, fname, canonical_name: canonical_fname, basis }
    }
}

/// Possible items to test against
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckBasis {
    /// Check against Musl's math sources.
    Musl,
}

/// A trait to implement on any output type so we can verify it in a generic way.
pub trait CheckOutput<Input>: Sized {
    /// Validate `self` (actual) and `expected` are the same.
    ///
    /// `input` is only used here for error messages.
    fn validate<'a>(self, expected: Self, input: Input, ctx: &CheckCtx) -> TestResult;
}

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

impl<T1, T2, T3> TupleCall<fn(T1, &mut T2, &mut T3)> for (T1,)
where
    T1: fmt::Debug,
    T2: fmt::Debug + Default,
    T3: fmt::Debug + Default,
{
    type Output = (T2, T3);

    fn call(self, f: fn(T1, &mut T2, &mut T3)) -> Self::Output {
        let mut t2 = T2::default();
        let mut t3 = T3::default();
        f(self.0, &mut t2, &mut t3);
        (t2, t3)
    }
}

// Implement for floats
impl<F, Input> CheckOutput<Input> for F
where
    F: Float + Hex,
    Input: Hex + fmt::Debug,
    u32: TryFrom<F::SignedInt, Error: fmt::Debug>,
    SpecialCase: MaybeOverride<Input>,
{
    fn validate<'a>(self, expected: Self, input: Input, ctx: &CheckCtx) -> TestResult {
        // Create a wrapper function so we only need to `.with_context` once.
        let inner = || -> TestResult {
            let mut allowed_ulp = ctx.ulp;

            // If the tested function requires a nonstandard test, run it here.
            if let Some(res) =
                SpecialCase::check_float(input, self, expected, &mut allowed_ulp, ctx)
            {
                return res;
            }

            // Check when both are NaNs
            if self.is_nan() && expected.is_nan() {
                // By default, NaNs have nothing special to check.
                return Ok(());
            } else if self.is_nan() || expected.is_nan() {
                // Check when only one is a NaN
                bail!("real value != NaN")
            }

            // Make sure that the signs are the same before checing ULP to avoid wraparound
            let act_sig = self.signum();
            let exp_sig = expected.signum();
            ensure!(act_sig == exp_sig, "mismatched signs {act_sig} {exp_sig}");

            if self.is_infinite() ^ expected.is_infinite() {
                bail!("mismatched infinities");
            }

            let act_bits = self.to_bits().signed();
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
                \n    actual:   {self:<22?} {actbits}\
                ",
                actbits = self.hex(),
                expbits = expected.hex(),
                ibits = input.hex(),
            )
        })
    }
}

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
