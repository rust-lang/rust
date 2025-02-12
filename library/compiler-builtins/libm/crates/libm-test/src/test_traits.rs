//! Traits related to testing.
//!
//! There are two main traits in this module:
//!
//! - `TupleCall`: implemented on tuples to allow calling them as function arguments.
//! - `CheckOutput`: implemented on anything that is an output type for validation against an
//!   expected value.

use std::panic::{RefUnwindSafe, UnwindSafe};
use std::{fmt, panic};

use anyhow::{Context, anyhow, bail, ensure};
use libm::support::Hexf;

use crate::precision::CheckAction;
use crate::{
    CheckBasis, CheckCtx, Float, GeneratorKind, Int, MaybeOverride, SpecialCase, TestResult,
};

/// Trait for calling a function with a tuple as arguments.
///
/// Implemented on the tuple with the function signature as the generic (so we can use the same
/// tuple for multiple signatures).
pub trait TupleCall<Func>: fmt::Debug {
    type Output;
    fn call(self, f: Func) -> Self::Output;

    /// Intercept panics and print the input to stderr before continuing.
    fn call_intercept_panics(self, f: Func) -> Self::Output
    where
        Self: RefUnwindSafe + Copy,
        Func: UnwindSafe,
    {
        let res = panic::catch_unwind(|| self.call(f));
        match res {
            Ok(v) => v,
            Err(e) => {
                eprintln!("panic with the following input: {self:?}");
                panic::resume_unwind(e)
            }
        }
    }
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
    /// Hex integer syntax.
    fn hex(self) -> String;
    /// Hex float syntax.
    fn hexf(self) -> String;
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

    fn hexf(self) -> String {
        format!("({},)", self.0.hexf())
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

    fn hexf(self) -> String {
        format!("({}, {})", self.0.hexf(), self.1.hexf())
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

    fn hexf(self) -> String {
        format!("({}, {}, {})", self.0.hexf(), self.1.hexf(), self.2.hexf())
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

                fn hexf(self) -> String {
                    String::new()
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
    let (result, xfail_msg) = match SpecialCase::check_int(input, actual, expected, ctx) {
        // `require_biteq` forbids overrides.
        _ if ctx.gen_kind == GeneratorKind::List => (actual == expected, None),
        CheckAction::AssertSuccess => (actual == expected, None),
        CheckAction::AssertFailure(msg) => (actual != expected, Some(msg)),
        CheckAction::Custom(res) => return res,
        CheckAction::Skip => return Ok(()),
        CheckAction::AssertWithUlp(_) => panic!("ulp has no meaning for integer checks"),
    };

    let make_xfail_msg = || match xfail_msg {
        Some(m) => format!(
            "expected failure but test passed. Does an XFAIL need to be updated?\n\
            failed at: {m}",
        ),
        None => String::new(),
    };

    anyhow::ensure!(
        result,
        "\
        \n    input:    {input:?} {ibits}\
        \n    expected: {expected:<22?} {expbits}\
        \n    actual:   {actual:<22?} {actbits}\
        \n    {msg}\
        ",
        actbits = actual.hex(),
        expbits = expected.hex(),
        ibits = input.hex(),
        msg = make_xfail_msg()
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

                fn hexf(self) -> String {
                    format!("{}", Hexf(self))
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
    let mut assert_failure_msg = None;

    // Create a wrapper function so we only need to `.with_context` once.
    let mut inner = || -> TestResult {
        let mut allowed_ulp = ctx.ulp;

        // Forbid overrides if the items came from an explicit list, as long as we are checking
        // against either MPFR or the result itself.
        let require_biteq = ctx.gen_kind == GeneratorKind::List && ctx.basis != CheckBasis::Musl;

        match SpecialCase::check_float(input, actual, expected, ctx) {
            _ if require_biteq => (),
            CheckAction::AssertSuccess => (),
            CheckAction::AssertFailure(msg) => assert_failure_msg = Some(msg),
            CheckAction::Custom(res) => return res,
            CheckAction::Skip => return Ok(()),
            CheckAction::AssertWithUlp(ulp_override) => allowed_ulp = ulp_override,
        };

        // Check when both are NaNs
        if actual.is_nan() && expected.is_nan() {
            if require_biteq && ctx.basis == CheckBasis::None {
                ensure!(actual.to_bits() == expected.to_bits(), "mismatched NaN bitpatterns");
            }
            // By default, NaNs have nothing special to check.
            return Ok(());
        } else if actual.is_nan() || expected.is_nan() {
            // Check when only one is a NaN
            bail!("real value != NaN")
        }

        // Make sure that the signs are the same before checing ULP to avoid wraparound
        let act_sig = actual.signum();
        let exp_sig = expected.signum();
        ensure!(act_sig == exp_sig, "mismatched signs {act_sig:?} {exp_sig:?}");

        if actual.is_infinite() ^ expected.is_infinite() {
            bail!("mismatched infinities");
        }

        let act_bits = actual.to_bits().signed();
        let exp_bits = expected.to_bits().signed();

        let ulp_diff = act_bits.checked_sub(exp_bits).unwrap().abs();

        let ulp_u32 = u32::try_from(ulp_diff)
            .map_err(|e| anyhow!("{e:?}: ulp of {ulp_diff} exceeds u32::MAX"))?;

        ensure!(ulp_u32 <= allowed_ulp, "ulp {ulp_diff} > {allowed_ulp}",);

        Ok(())
    };

    let mut res = inner();

    if let Some(msg) = assert_failure_msg {
        // Invert `Ok` and `Err` if the test is an xfail.
        if res.is_ok() {
            let e = anyhow!(
                "expected failure but test passed. Does an XFAIL need to be updated?\n\
                failed at: {msg}",
            );
            res = Err(e)
        } else {
            res = Ok(())
        }
    }

    res.with_context(|| {
        format!(
            "\
            \n    input:    {input:?}\
            \n    as hex:   {ihex}\
            \n    as bits:  {ibits}\
            \n    expected: {expected:<22?} {exphex} {expbits}\
            \n    actual:   {actual:<22?} {acthex} {actbits}\
            ",
            ihex = input.hexf(),
            ibits = input.hex(),
            exphex = expected.hexf(),
            expbits = expected.hex(),
            actbits = actual.hex(),
            acthex = actual.hexf(),
        )
    })
}

impl_float!(f32, f64);

#[cfg(f16_enabled)]
impl_float!(f16);

#[cfg(f128_enabled)]
impl_float!(f128);

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
                            \n    as hex:   {ihex}\
                            \n    as bits:  {ibits}\
                            \n    expected: {expected:?} {expbits}\
                            \n    actual:   {self:?} {actbits}\
                            ",
                            ihex = input.hexf(),
                            ibits = input.hex(),
                            expbits = expected.hex(),
                            actbits = self.hex(),
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
