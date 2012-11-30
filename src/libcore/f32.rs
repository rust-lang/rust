// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

//! Operations and constants for `f32`

pub use cmath::c_float_utils::*;
pub use cmath::c_float_targ_consts::*;

// These are not defined inside consts:: for consistency with
// the integer types

pub const NaN: f32 = 0.0_f32/0.0_f32;

pub const infinity: f32 = 1.0_f32/0.0_f32;

pub const neg_infinity: f32 = -1.0_f32/0.0_f32;

pub pure fn is_NaN(f: f32) -> bool { f != f }

pub pure fn add(x: f32, y: f32) -> f32 { return x + y; }

pub pure fn sub(x: f32, y: f32) -> f32 { return x - y; }

pub pure fn mul(x: f32, y: f32) -> f32 { return x * y; }

pub pure fn div(x: f32, y: f32) -> f32 { return x / y; }

pub pure fn rem(x: f32, y: f32) -> f32 { return x % y; }

pub pure fn lt(x: f32, y: f32) -> bool { return x < y; }

pub pure fn le(x: f32, y: f32) -> bool { return x <= y; }

pub pure fn eq(x: f32, y: f32) -> bool { return x == y; }

pub pure fn ne(x: f32, y: f32) -> bool { return x != y; }

pub pure fn ge(x: f32, y: f32) -> bool { return x >= y; }

pub pure fn gt(x: f32, y: f32) -> bool { return x > y; }

// FIXME (#1999): replace the predicates below with llvm intrinsics or
// calls to the libmath macros in the rust runtime for performance.

/// Returns true if `x` is a positive number, including +0.0f320 and +Infinity
pub pure fn is_positive(x: f32) -> bool
    { return x > 0.0f32 || (1.0f32/x) == infinity; }

/// Returns true if `x` is a negative number, including -0.0f320 and -Infinity
pub pure fn is_negative(x: f32) -> bool
    { return x < 0.0f32 || (1.0f32/x) == neg_infinity; }

/**
 * Returns true if `x` is a negative number, including -0.0f320 and -Infinity
 *
 * This is the same as `f32::is_negative`.
 */
pub pure fn is_nonpositive(x: f32) -> bool {
  return x < 0.0f32 || (1.0f32/x) == neg_infinity;
}

/**
 * Returns true if `x` is a positive number, including +0.0f320 and +Infinity
 *
 * This is the same as `f32::is_positive`.)
 */
pub pure fn is_nonnegative(x: f32) -> bool {
  return x > 0.0f32 || (1.0f32/x) == infinity;
}

/// Returns true if `x` is a zero number (positive or negative zero)
pub pure fn is_zero(x: f32) -> bool {
    return x == 0.0f32 || x == -0.0f32;
}

/// Returns true if `x`is an infinite number
pub pure fn is_infinite(x: f32) -> bool {
    return x == infinity || x == neg_infinity;
}

/// Returns true if `x`is a finite number
pub pure fn is_finite(x: f32) -> bool {
    return !(is_NaN(x) || is_infinite(x));
}

// FIXME (#1999): add is_normal, is_subnormal, and fpclassify.

/* Module: consts */
pub mod consts {
    // FIXME (requires Issue #1433 to fix): replace with mathematical
    // constants from cmath.
    /// Archimedes' constant
    pub const pi: f32 = 3.14159265358979323846264338327950288_f32;

    /// pi/2.0
    pub const frac_pi_2: f32 = 1.57079632679489661923132169163975144_f32;

    /// pi/4.0
    pub const frac_pi_4: f32 = 0.785398163397448309615660845819875721_f32;

    /// 1.0/pi
    pub const frac_1_pi: f32 = 0.318309886183790671537767526745028724_f32;

    /// 2.0/pi
    pub const frac_2_pi: f32 = 0.636619772367581343075535053490057448_f32;

    /// 2.0/sqrt(pi)
    pub const frac_2_sqrtpi: f32 = 1.12837916709551257389615890312154517_f32;

    /// sqrt(2.0)
    pub const sqrt2: f32 = 1.41421356237309504880168872420969808_f32;

    /// 1.0/sqrt(2.0)
    pub const frac_1_sqrt2: f32 = 0.707106781186547524400844362104849039_f32;

    /// Euler's number
    pub const e: f32 = 2.71828182845904523536028747135266250_f32;

    /// log2(e)
    pub const log2_e: f32 = 1.44269504088896340735992468100189214_f32;

    /// log10(e)
    pub const log10_e: f32 = 0.434294481903251827651128918916605082_f32;

    /// ln(2.0)
    pub const ln_2: f32 = 0.693147180559945309417232121458176568_f32;

    /// ln(10.0)
    pub const ln_10: f32 = 2.30258509299404568401799145468436421_f32;
}

pub pure fn signbit(x: f32) -> int {
    if is_negative(x) { return 1; } else { return 0; }
}

pub pure fn logarithm(n: f32, b: f32) -> f32 {
    return log2(n) / log2(b);
}

#[cfg(notest)]
impl f32 : cmp::Eq {
    pure fn eq(&self, other: &f32) -> bool { (*self) == (*other) }
    pure fn ne(&self, other: &f32) -> bool { (*self) != (*other) }
}

#[cfg(notest)]
impl f32 : cmp::Ord {
    pure fn lt(&self, other: &f32) -> bool { (*self) < (*other) }
    pure fn le(&self, other: &f32) -> bool { (*self) <= (*other) }
    pure fn ge(&self, other: &f32) -> bool { (*self) >= (*other) }
    pure fn gt(&self, other: &f32) -> bool { (*self) > (*other) }
}

impl f32: num::Num {
    pure fn add(other: &f32)    -> f32 { return self + *other; }
    pure fn sub(other: &f32)    -> f32 { return self - *other; }
    pure fn mul(other: &f32)    -> f32 { return self * *other; }
    pure fn div(other: &f32)    -> f32 { return self / *other; }
    pure fn modulo(other: &f32) -> f32 { return self % *other; }
    pure fn neg()                -> f32 { return -self;        }

    pure fn to_int()         -> int { return self as int; }
    static pure fn from_int(n: int) -> f32 { return n as f32;    }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
