//! Traits and operations related to bounds of a function.

use std::fmt;
use std::ops::Bound;

use libm::support::Int;

use crate::{BaseName, Float, FloatExt, Identifier};

/// Representation of a single dimension of a function's domain.
#[derive(Clone, Debug)]
pub struct Domain<T> {
    /// Start of the region for which a function is defined (ignoring poles).
    pub start: Bound<T>,
    /// Endof the region for which a function is defined (ignoring poles).
    pub end: Bound<T>,
    /// Additional points to check closer around. These can be e.g. undefined asymptotes or
    /// inflection points.
    pub check_points: Option<fn() -> BoxIter<T>>,
}

type BoxIter<T> = Box<dyn Iterator<Item = T>>;

impl<F: FloatExt> Domain<F> {
    /// The start of this domain, saturating at negative infinity.
    pub fn range_start(&self) -> F {
        match self.start {
            Bound::Included(v) => v,
            Bound::Excluded(v) => v.next_up(),
            Bound::Unbounded => F::NEG_INFINITY,
        }
    }

    /// The end of this domain, saturating at infinity.
    pub fn range_end(&self) -> F {
        match self.end {
            Bound::Included(v) => v,
            Bound::Excluded(v) => v.next_down(),
            Bound::Unbounded => F::INFINITY,
        }
    }
}

/// A value that may be any float type or any integer type.
#[derive(Clone, Debug)]
pub enum EitherPrim<F, I> {
    Float(F),
    Int(I),
}

impl<F: fmt::Debug, I: fmt::Debug> EitherPrim<F, I> {
    pub fn unwrap_float(self) -> F {
        match self {
            EitherPrim::Float(f) => f,
            EitherPrim::Int(_) => panic!("expected float; got {self:?}"),
        }
    }

    pub fn unwrap_int(self) -> I {
        match self {
            EitherPrim::Float(_) => panic!("expected int; got {self:?}"),
            EitherPrim::Int(i) => i,
        }
    }
}

/// Convenience 1-dimensional float domains.
impl<F: Float> Domain<F> {
    /// x ∈ ℝ
    const UNBOUNDED: Self =
        Self { start: Bound::Unbounded, end: Bound::Unbounded, check_points: None };

    /// x ∈ ℝ >= 0
    const POSITIVE: Self =
        Self { start: Bound::Included(F::ZERO), end: Bound::Unbounded, check_points: None };

    /// x ∈ ℝ > 0
    const STRICTLY_POSITIVE: Self =
        Self { start: Bound::Excluded(F::ZERO), end: Bound::Unbounded, check_points: None };

    /// Wrap in the float variant of [`EitherPrim`].
    const fn into_prim_float<I>(self) -> EitherPrim<Self, Domain<I>> {
        EitherPrim::Float(self)
    }
}

/// Convenience 1-dimensional integer domains.
impl<I: Int> Domain<I> {
    /// x ∈ ℝ
    const UNBOUNDED_INT: Self =
        Self { start: Bound::Unbounded, end: Bound::Unbounded, check_points: None };

    /// Wrap in the int variant of [`EitherPrim`].
    const fn into_prim_int<F>(self) -> EitherPrim<Domain<F>, Self> {
        EitherPrim::Int(self)
    }
}

/// Multidimensional domains, represented as an array of 1-D domains.
impl<F: Float, I: Int> EitherPrim<Domain<F>, Domain<I>> {
    /// x ∈ ℝ
    const UNBOUNDED1: [Self; 1] =
        [Domain { start: Bound::Unbounded, end: Bound::Unbounded, check_points: None }
            .into_prim_float()];

    /// {x1, x2} ∈ ℝ
    const UNBOUNDED2: [Self; 2] =
        [Domain::UNBOUNDED.into_prim_float(), Domain::UNBOUNDED.into_prim_float()];

    /// {x1, x2, x3} ∈ ℝ
    const UNBOUNDED3: [Self; 3] = [
        Domain::UNBOUNDED.into_prim_float(),
        Domain::UNBOUNDED.into_prim_float(),
        Domain::UNBOUNDED.into_prim_float(),
    ];

    /// {x1, x2} ∈ ℝ, one float and one int
    const UNBOUNDED_F_I: [Self; 2] =
        [Domain::UNBOUNDED.into_prim_float(), Domain::UNBOUNDED_INT.into_prim_int()];

    /// x ∈ ℝ >= 0
    const POSITIVE: [Self; 1] = [Domain::POSITIVE.into_prim_float()];

    /// x ∈ ℝ > 0
    const STRICTLY_POSITIVE: [Self; 1] = [Domain::STRICTLY_POSITIVE.into_prim_float()];

    /// Used for versions of `asin` and `acos`.
    const INVERSE_TRIG_PERIODIC: [Self; 1] = [Domain {
        start: Bound::Included(F::NEG_ONE),
        end: Bound::Included(F::ONE),
        check_points: None,
    }
    .into_prim_float()];

    /// Domain for `acosh`
    const ACOSH: [Self; 1] =
        [Domain { start: Bound::Included(F::ONE), end: Bound::Unbounded, check_points: None }
            .into_prim_float()];

    /// Domain for `atanh`
    const ATANH: [Self; 1] = [Domain {
        start: Bound::Excluded(F::NEG_ONE),
        end: Bound::Excluded(F::ONE),
        check_points: None,
    }
    .into_prim_float()];

    /// Domain for `sin`, `cos`, and `tan`
    const TRIG: [Self; 1] = [Domain {
        // Trig functions have special behavior at fractions of π.
        check_points: Some(|| Box::new([-F::PI, -F::FRAC_PI_2, F::FRAC_PI_2, F::PI].into_iter())),
        ..Domain::UNBOUNDED
    }
    .into_prim_float()];

    /// Domain for `log` in various bases
    const LOG: [Self; 1] = Self::STRICTLY_POSITIVE;

    /// Domain for `log1p` i.e. `log(1 + x)`
    const LOG1P: [Self; 1] =
        [Domain { start: Bound::Excluded(F::NEG_ONE), end: Bound::Unbounded, check_points: None }
            .into_prim_float()];

    /// Domain for `sqrt`
    const SQRT: [Self; 1] = Self::POSITIVE;

    /// Domain for `gamma`
    const GAMMA: [Self; 1] = [Domain {
        check_points: Some(|| {
            // Negative integers are asymptotes
            Box::new((0..u8::MAX).map(|scale| {
                let mut base = F::ZERO;
                for _ in 0..scale {
                    base = base - F::ONE;
                }
                base
            }))
        }),
        // Whether or not gamma is defined for negative numbers is implementation dependent
        ..Domain::UNBOUNDED
    }
    .into_prim_float()];

    /// Domain for `loggamma`
    const LGAMMA: [Self; 1] = Self::STRICTLY_POSITIVE;

    /// Domain for `jn` and `yn`.
    // FIXME: the domain should provide some sort of "reasonable range" so we don't actually test
    // the entire system unbounded.
    const BESSEL_N: [Self; 2] =
        [Domain::UNBOUNDED_INT.into_prim_int(), Domain::UNBOUNDED.into_prim_float()];
}

/// Get the domain for a given function.
pub fn get_domain<F: Float, I: Int>(
    id: Identifier,
    argnum: usize,
) -> EitherPrim<Domain<F>, Domain<I>> {
    let x = match id.base_name() {
        BaseName::Acos => &EitherPrim::INVERSE_TRIG_PERIODIC[..],
        BaseName::Acosh => &EitherPrim::ACOSH[..],
        BaseName::Asin => &EitherPrim::INVERSE_TRIG_PERIODIC[..],
        BaseName::Asinh => &EitherPrim::UNBOUNDED1[..],
        BaseName::Atan => &EitherPrim::UNBOUNDED1[..],
        BaseName::Atan2 => &EitherPrim::UNBOUNDED2[..],
        BaseName::Cbrt => &EitherPrim::UNBOUNDED1[..],
        BaseName::Atanh => &EitherPrim::ATANH[..],
        BaseName::Ceil => &EitherPrim::UNBOUNDED1[..],
        BaseName::Cosh => &EitherPrim::UNBOUNDED1[..],
        BaseName::Copysign => &EitherPrim::UNBOUNDED2[..],
        BaseName::Cos => &EitherPrim::TRIG[..],
        BaseName::Exp => &EitherPrim::UNBOUNDED1[..],
        BaseName::Erf => &EitherPrim::UNBOUNDED1[..],
        BaseName::Erfc => &EitherPrim::UNBOUNDED1[..],
        BaseName::Expm1 => &EitherPrim::UNBOUNDED1[..],
        BaseName::Exp10 => &EitherPrim::UNBOUNDED1[..],
        BaseName::Exp2 => &EitherPrim::UNBOUNDED1[..],
        BaseName::Frexp => &EitherPrim::UNBOUNDED1[..],
        BaseName::Fabs => &EitherPrim::UNBOUNDED1[..],
        BaseName::Fdim => &EitherPrim::UNBOUNDED2[..],
        BaseName::Floor => &EitherPrim::UNBOUNDED1[..],
        BaseName::Fma => &EitherPrim::UNBOUNDED3[..],
        BaseName::Fmax => &EitherPrim::UNBOUNDED2[..],
        BaseName::Fmin => &EitherPrim::UNBOUNDED2[..],
        BaseName::Fmod => &EitherPrim::UNBOUNDED2[..],
        BaseName::Hypot => &EitherPrim::UNBOUNDED2[..],
        BaseName::Ilogb => &EitherPrim::UNBOUNDED1[..],
        BaseName::J0 => &EitherPrim::UNBOUNDED1[..],
        BaseName::J1 => &EitherPrim::UNBOUNDED1[..],
        BaseName::Jn => &EitherPrim::BESSEL_N[..],
        BaseName::Ldexp => &EitherPrim::UNBOUNDED_F_I[..],
        BaseName::Lgamma => &EitherPrim::LGAMMA[..],
        BaseName::LgammaR => &EitherPrim::LGAMMA[..],
        BaseName::Log => &EitherPrim::LOG[..],
        BaseName::Log10 => &EitherPrim::LOG[..],
        BaseName::Log1p => &EitherPrim::LOG1P[..],
        BaseName::Log2 => &EitherPrim::LOG[..],
        BaseName::Modf => &EitherPrim::UNBOUNDED1[..],
        BaseName::Nextafter => &EitherPrim::UNBOUNDED2[..],
        BaseName::Pow => &EitherPrim::UNBOUNDED2[..],
        BaseName::Remainder => &EitherPrim::UNBOUNDED2[..],
        BaseName::Remquo => &EitherPrim::UNBOUNDED2[..],
        BaseName::Rint => &EitherPrim::UNBOUNDED1[..],
        BaseName::Round => &EitherPrim::UNBOUNDED1[..],
        BaseName::Scalbn => &EitherPrim::UNBOUNDED_F_I[..],
        BaseName::Sin => &EitherPrim::TRIG[..],
        BaseName::Sincos => &EitherPrim::TRIG[..],
        BaseName::Sinh => &EitherPrim::UNBOUNDED1[..],
        BaseName::Sqrt => &EitherPrim::SQRT[..],
        BaseName::Tan => &EitherPrim::TRIG[..],
        BaseName::Tanh => &EitherPrim::UNBOUNDED1[..],
        BaseName::Tgamma => &EitherPrim::GAMMA[..],
        BaseName::Trunc => &EitherPrim::UNBOUNDED1[..],
        BaseName::Y0 => &EitherPrim::UNBOUNDED1[..],
        BaseName::Y1 => &EitherPrim::UNBOUNDED1[..],
        BaseName::Yn => &EitherPrim::BESSEL_N[..],
    };

    x[argnum].clone()
}
