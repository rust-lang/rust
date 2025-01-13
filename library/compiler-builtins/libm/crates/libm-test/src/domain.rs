//! Traits and operations related to bounds of a function.

use std::fmt;
use std::ops::{self, Bound};

use crate::{Float, FloatExt};

/// Representation of a function's domain.
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

impl<F: Float> Domain<F> {
    /// x ∈ ℝ
    pub const UNBOUNDED: Self =
        Self { start: Bound::Unbounded, end: Bound::Unbounded, check_points: None };

    /// x ∈ ℝ >= 0
    pub const POSITIVE: Self =
        Self { start: Bound::Included(F::ZERO), end: Bound::Unbounded, check_points: None };

    /// x ∈ ℝ > 0
    pub const STRICTLY_POSITIVE: Self =
        Self { start: Bound::Excluded(F::ZERO), end: Bound::Unbounded, check_points: None };

    /// Used for versions of `asin` and `acos`.
    pub const INVERSE_TRIG_PERIODIC: Self = Self {
        start: Bound::Included(F::NEG_ONE),
        end: Bound::Included(F::ONE),
        check_points: None,
    };

    /// Domain for `acosh`
    pub const ACOSH: Self =
        Self { start: Bound::Included(F::ONE), end: Bound::Unbounded, check_points: None };

    /// Domain for `atanh`
    pub const ATANH: Self = Self {
        start: Bound::Excluded(F::NEG_ONE),
        end: Bound::Excluded(F::ONE),
        check_points: None,
    };

    /// Domain for `sin`, `cos`, and `tan`
    pub const TRIG: Self = Self {
        // TODO
        check_points: Some(|| Box::new([-F::PI, -F::FRAC_PI_2, F::FRAC_PI_2, F::PI].into_iter())),
        ..Self::UNBOUNDED
    };

    /// Domain for `log` in various bases
    pub const LOG: Self = Self::STRICTLY_POSITIVE;

    /// Domain for `log1p` i.e. `log(1 + x)`
    pub const LOG1P: Self =
        Self { start: Bound::Excluded(F::NEG_ONE), end: Bound::Unbounded, check_points: None };

    /// Domain for `sqrt`
    pub const SQRT: Self = Self::POSITIVE;

    /// Domain for `gamma`
    pub const GAMMA: Self = Self {
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
        ..Self::UNBOUNDED
    };

    /// Domain for `loggamma`
    pub const LGAMMA: Self = Self::STRICTLY_POSITIVE;
}

/// Implement on `op::*` types to indicate how they are bounded.
pub trait HasDomain<T>
where
    T: Copy + fmt::Debug + ops::Add<Output = T> + ops::Sub<Output = T> + PartialOrd + 'static,
{
    const DOMAIN: Domain<T>;
}

/// Implement [`HasDomain`] for both the `f32` and `f64` variants of a function.
macro_rules! impl_has_domain {
    ($($fn_name:ident => $domain:expr;)*) => {
        paste::paste! {
            $(
                // Implement for f64 functions
                impl HasDomain<f64> for $crate::op::$fn_name::Routine {
                    const DOMAIN: Domain<f64> = Domain::<f64>::$domain;
                }

                // Implement for f32 functions
                impl HasDomain<f32> for $crate::op::[< $fn_name f >]::Routine {
                    const DOMAIN: Domain<f32> = Domain::<f32>::$domain;
                }
            )*
        }
    };
}

// Tie functions together with their domains.
impl_has_domain! {
    acos => INVERSE_TRIG_PERIODIC;
    acosh => ACOSH;
    asin => INVERSE_TRIG_PERIODIC;
    asinh => UNBOUNDED;
    atan => UNBOUNDED;
    atanh => ATANH;
    cbrt => UNBOUNDED;
    ceil => UNBOUNDED;
    cos => TRIG;
    cosh => UNBOUNDED;
    erf => UNBOUNDED;
    erfc => UNBOUNDED;
    exp => UNBOUNDED;
    exp10 => UNBOUNDED;
    exp2 => UNBOUNDED;
    expm1 => UNBOUNDED;
    fabs => UNBOUNDED;
    floor => UNBOUNDED;
    frexp => UNBOUNDED;
    ilogb => UNBOUNDED;
    j0 => UNBOUNDED;
    j1 => UNBOUNDED;
    lgamma => LGAMMA;
    log => LOG;
    log10 => LOG;
    log1p => LOG1P;
    log2 => LOG;
    modf => UNBOUNDED;
    rint => UNBOUNDED;
    round => UNBOUNDED;
    sin => TRIG;
    sincos => TRIG;
    sinh => UNBOUNDED;
    sqrt => SQRT;
    tan => TRIG;
    tanh => UNBOUNDED;
    tgamma => GAMMA;
    trunc => UNBOUNDED;
    y0 => UNBOUNDED;
    y1 => UNBOUNDED;
}

/* Manual implementations, these functions don't follow `foo`->`foof` naming */

impl HasDomain<f32> for crate::op::lgammaf_r::Routine {
    const DOMAIN: Domain<f32> = Domain::<f32>::LGAMMA;
}

impl HasDomain<f64> for crate::op::lgamma_r::Routine {
    const DOMAIN: Domain<f64> = Domain::<f64>::LGAMMA;
}

/* Not all `f16` and `f128` functions exist yet so we can't easily use the macros. */

#[cfg(f16_enabled)]
impl HasDomain<f16> for crate::op::fabsf16::Routine {
    const DOMAIN: Domain<f16> = Domain::<f16>::UNBOUNDED;
}

#[cfg(f128_enabled)]
impl HasDomain<f128> for crate::op::fabsf128::Routine {
    const DOMAIN: Domain<f128> = Domain::<f128>::UNBOUNDED;
}

#[cfg(f16_enabled)]
impl HasDomain<f16> for crate::op::truncf16::Routine {
    const DOMAIN: Domain<f16> = Domain::<f16>::UNBOUNDED;
}

#[cfg(f128_enabled)]
impl HasDomain<f128> for crate::op::truncf128::Routine {
    const DOMAIN: Domain<f128> = Domain::<f128>::UNBOUNDED;
}
