use crate::{Category, ExpInt, Float, FloatConvert, Round, ParseError, Status, StatusAnd};
use crate::ieee;

use std::cmp::Ordering;
use std::fmt;
use std::ops::Neg;

#[must_use]
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct DoubleFloat<F>(F, F);
pub type DoubleDouble = DoubleFloat<ieee::Double>;

// These are legacy semantics for the Fallback, inaccurate implementation of
// IBM double-double, if the accurate DoubleDouble doesn't handle the
// operation. It's equivalent to having an IEEE number with consecutive 106
// bits of mantissa and 11 bits of exponent.
//
// It's not equivalent to IBM double-double. For example, a legit IBM
// double-double, 1 + epsilon:
//
//   1 + epsilon = 1 + (1 >> 1076)
//
// is not representable by a consecutive 106 bits of mantissa.
//
// Currently, these semantics are used in the following way:
//
//   DoubleDouble -> (Double, Double) ->
//   DoubleDouble's Fallback -> IEEE operations
//
// FIXME: Implement all operations in DoubleDouble, and delete these
// semantics.
// FIXME(eddyb) This shouldn't need to be `pub`, it's only used in bounds.
pub struct FallbackS<F>(F);
type Fallback<F> = ieee::IeeeFloat<FallbackS<F>>;
impl<F: Float> ieee::Semantics for FallbackS<F> {
    // Forbid any conversion to/from bits.
    const BITS: usize = 0;
    const PRECISION: usize = F::PRECISION * 2;
    const MAX_EXP: ExpInt = F::MAX_EXP as ExpInt;
    const MIN_EXP: ExpInt = F::MIN_EXP as ExpInt + F::PRECISION as ExpInt;
}

// Convert number to F. To avoid spurious underflows, we re-
// normalize against the F exponent range first, and only *then*
// truncate the mantissa. The result of that second conversion
// may be inexact, but should never underflow.
// FIXME(eddyb) This shouldn't need to be `pub`, it's only used in bounds.
pub struct FallbackExtendedS<F>(F);
type FallbackExtended<F> = ieee::IeeeFloat<FallbackExtendedS<F>>;
impl<F: Float> ieee::Semantics for FallbackExtendedS<F> {
    // Forbid any conversion to/from bits.
    const BITS: usize = 0;
    const PRECISION: usize = Fallback::<F>::PRECISION;
    const MAX_EXP: ExpInt = F::MAX_EXP as ExpInt;
}

impl<F: Float> From<Fallback<F>> for DoubleFloat<F>
where
    F: FloatConvert<FallbackExtended<F>>,
    FallbackExtended<F>: FloatConvert<F>,
{
    fn from(x: Fallback<F>) -> Self {
        let mut status;
        let mut loses_info = false;

        let extended: FallbackExtended<F> = unpack!(status=, x.convert(&mut loses_info));
        assert_eq!((status, loses_info), (Status::OK, false));

        let a = unpack!(status=, extended.convert(&mut loses_info));
        assert_eq!(status - Status::INEXACT, Status::OK);

        // If conversion was exact or resulted in a special case, we're done;
        // just set the second double to zero. Otherwise, re-convert back to
        // the extended format and compute the difference. This now should
        // convert exactly to double.
        let b = if a.is_finite_non_zero() && loses_info {
            let u: FallbackExtended<F> = unpack!(status=, a.convert(&mut loses_info));
            assert_eq!((status, loses_info), (Status::OK, false));
            let v = unpack!(status=, extended - u);
            assert_eq!(status, Status::OK);
            let v = unpack!(status=, v.convert(&mut loses_info));
            assert_eq!((status, loses_info), (Status::OK, false));
            v
        } else {
            F::ZERO
        };

        DoubleFloat(a, b)
    }
}

impl<F: FloatConvert<Self>> From<DoubleFloat<F>> for Fallback<F> {
    fn from(DoubleFloat(a, b): DoubleFloat<F>) -> Self {
        let mut status;
        let mut loses_info = false;

        // Get the first F and convert to our format.
        let a = unpack!(status=, a.convert(&mut loses_info));
        assert_eq!((status, loses_info), (Status::OK, false));

        // Unless we have a special case, add in second F.
        if a.is_finite_non_zero() {
            let b = unpack!(status=, b.convert(&mut loses_info));
            assert_eq!((status, loses_info), (Status::OK, false));

            (a + b).value
        } else {
            a
        }
    }
}

float_common_impls!(DoubleFloat<F>);

impl<F: Float> Neg for DoubleFloat<F> {
    type Output = Self;
    fn neg(self) -> Self {
        if self.1.is_finite_non_zero() {
            DoubleFloat(-self.0, -self.1)
        } else {
            DoubleFloat(-self.0, self.1)
        }
    }
}

impl<F: FloatConvert<Fallback<F>>> fmt::Display for DoubleFloat<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&Fallback::from(*self), f)
    }
}

impl<F: FloatConvert<Fallback<F>>> Float for DoubleFloat<F>
where
    Self: From<Fallback<F>>,
{
    const BITS: usize = F::BITS * 2;
    const PRECISION: usize = Fallback::<F>::PRECISION;
    const MAX_EXP: ExpInt = Fallback::<F>::MAX_EXP;
    const MIN_EXP: ExpInt = Fallback::<F>::MIN_EXP;

    const ZERO: Self = DoubleFloat(F::ZERO, F::ZERO);

    const INFINITY: Self = DoubleFloat(F::INFINITY, F::ZERO);

    // FIXME(eddyb) remove when qnan becomes const fn.
    const NAN: Self = DoubleFloat(F::NAN, F::ZERO);

    fn qnan(payload: Option<u128>) -> Self {
        DoubleFloat(F::qnan(payload), F::ZERO)
    }

    fn snan(payload: Option<u128>) -> Self {
        DoubleFloat(F::snan(payload), F::ZERO)
    }

    fn largest() -> Self {
        let status;
        let mut r = DoubleFloat(F::largest(), F::largest());
        r.1 = r.1.scalbn(-(F::PRECISION as ExpInt + 1));
        r.1 = unpack!(status=, r.1.next_down());
        assert_eq!(status, Status::OK);
        r
    }

    const SMALLEST: Self = DoubleFloat(F::SMALLEST, F::ZERO);

    fn smallest_normalized() -> Self {
        DoubleFloat(
            F::smallest_normalized().scalbn(F::PRECISION as ExpInt),
            F::ZERO,
        )
    }

    // Implement addition, subtraction, multiplication and division based on:
    // "Software for Doubled-Precision Floating-Point Computations",
    // by Seppo Linnainmaa, ACM TOMS vol 7 no 3, September 1981, pages 272-283.

    fn add_r(mut self, rhs: Self, round: Round) -> StatusAnd<Self> {
        match (self.category(), rhs.category()) {
            (Category::Infinity, Category::Infinity) => {
                if self.is_negative() != rhs.is_negative() {
                    Status::INVALID_OP.and(Self::NAN.copy_sign(self))
                } else {
                    Status::OK.and(self)
                }
            }

            (_, Category::Zero) |
            (Category::NaN, _) |
            (Category::Infinity, Category::Normal) => Status::OK.and(self),

            (Category::Zero, _) |
            (_, Category::NaN) |
            (_, Category::Infinity) => Status::OK.and(rhs),

            (Category::Normal, Category::Normal) => {
                let mut status = Status::OK;
                let (a, aa, c, cc) = (self.0, self.1, rhs.0, rhs.1);
                let mut z = a;
                z = unpack!(status|=, z.add_r(c, round));
                if !z.is_finite() {
                    if !z.is_infinite() {
                        return status.and(DoubleFloat(z, F::ZERO));
                    }
                    status = Status::OK;
                    let a_cmp_c = a.cmp_abs_normal(c);
                    z = cc;
                    z = unpack!(status|=, z.add_r(aa, round));
                    if a_cmp_c == Ordering::Greater {
                        // z = cc + aa + c + a;
                        z = unpack!(status|=, z.add_r(c, round));
                        z = unpack!(status|=, z.add_r(a, round));
                    } else {
                        // z = cc + aa + a + c;
                        z = unpack!(status|=, z.add_r(a, round));
                        z = unpack!(status|=, z.add_r(c, round));
                    }
                    if !z.is_finite() {
                        return status.and(DoubleFloat(z, F::ZERO));
                    }
                    self.0 = z;
                    let mut zz = aa;
                    zz = unpack!(status|=, zz.add_r(cc, round));
                    if a_cmp_c == Ordering::Greater {
                        // self.1 = a - z + c + zz;
                        self.1 = a;
                        self.1 = unpack!(status|=, self.1.sub_r(z, round));
                        self.1 = unpack!(status|=, self.1.add_r(c, round));
                        self.1 = unpack!(status|=, self.1.add_r(zz, round));
                    } else {
                        // self.1 = c - z + a + zz;
                        self.1 = c;
                        self.1 = unpack!(status|=, self.1.sub_r(z, round));
                        self.1 = unpack!(status|=, self.1.add_r(a, round));
                        self.1 = unpack!(status|=, self.1.add_r(zz, round));
                    }
                } else {
                    // q = a - z;
                    let mut q = a;
                    q = unpack!(status|=, q.sub_r(z, round));

                    // zz = q + c + (a - (q + z)) + aa + cc;
                    // Compute a - (q + z) as -((q + z) - a) to avoid temporary copies.
                    let mut zz = q;
                    zz = unpack!(status|=, zz.add_r(c, round));
                    q = unpack!(status|=, q.add_r(z, round));
                    q = unpack!(status|=, q.sub_r(a, round));
                    q = -q;
                    zz = unpack!(status|=, zz.add_r(q, round));
                    zz = unpack!(status|=, zz.add_r(aa, round));
                    zz = unpack!(status|=, zz.add_r(cc, round));
                    if zz.is_zero() && !zz.is_negative() {
                        return Status::OK.and(DoubleFloat(z, F::ZERO));
                    }
                    self.0 = z;
                    self.0 = unpack!(status|=, self.0.add_r(zz, round));
                    if !self.0.is_finite() {
                        self.1 = F::ZERO;
                        return status.and(self);
                    }
                    self.1 = z;
                    self.1 = unpack!(status|=, self.1.sub_r(self.0, round));
                    self.1 = unpack!(status|=, self.1.add_r(zz, round));
                }
                status.and(self)
            }
        }
    }

    fn mul_r(mut self, rhs: Self, round: Round) -> StatusAnd<Self> {
        // Interesting observation: For special categories, finding the lowest
        // common ancestor of the following layered graph gives the correct
        // return category:
        //
        //    NaN
        //   /   \
        // Zero  Inf
        //   \   /
        //   Normal
        //
        // e.g., NaN * NaN = NaN
        //      Zero * Inf = NaN
        //      Normal * Zero = Zero
        //      Normal * Inf = Inf
        match (self.category(), rhs.category()) {
            (Category::NaN, _) => Status::OK.and(self),

            (_, Category::NaN) => Status::OK.and(rhs),

            (Category::Zero, Category::Infinity) |
            (Category::Infinity, Category::Zero) => Status::OK.and(Self::NAN),

            (Category::Zero, _) |
            (Category::Infinity, _) => Status::OK.and(self),

            (_, Category::Zero) |
            (_, Category::Infinity) => Status::OK.and(rhs),

            (Category::Normal, Category::Normal) => {
                let mut status = Status::OK;
                let (a, b, c, d) = (self.0, self.1, rhs.0, rhs.1);
                // t = a * c
                let mut t = a;
                t = unpack!(status|=, t.mul_r(c, round));
                if !t.is_finite_non_zero() {
                    return status.and(DoubleFloat(t, F::ZERO));
                }

                // tau = fmsub(a, c, t), that is -fmadd(-a, c, t).
                let mut tau = a;
                tau = unpack!(status|=, tau.mul_add_r(c, -t, round));
                // v = a * d
                let mut v = a;
                v = unpack!(status|=, v.mul_r(d, round));
                // w = b * c
                let mut w = b;
                w = unpack!(status|=, w.mul_r(c, round));
                v = unpack!(status|=, v.add_r(w, round));
                // tau += v + w
                tau = unpack!(status|=, tau.add_r(v, round));
                // u = t + tau
                let mut u = t;
                u = unpack!(status|=, u.add_r(tau, round));

                self.0 = u;
                if !u.is_finite() {
                    self.1 = F::ZERO;
                } else {
                    // self.1 = (t - u) + tau
                    t = unpack!(status|=, t.sub_r(u, round));
                    t = unpack!(status|=, t.add_r(tau, round));
                    self.1 = t;
                }
                status.and(self)
            }
        }
    }

    fn mul_add_r(self, multiplicand: Self, addend: Self, round: Round) -> StatusAnd<Self> {
        Fallback::from(self)
            .mul_add_r(Fallback::from(multiplicand), Fallback::from(addend), round)
            .map(Self::from)
    }

    fn div_r(self, rhs: Self, round: Round) -> StatusAnd<Self> {
        Fallback::from(self).div_r(Fallback::from(rhs), round).map(
            Self::from,
        )
    }

    fn c_fmod(self, rhs: Self) -> StatusAnd<Self> {
        Fallback::from(self).c_fmod(Fallback::from(rhs)).map(
            Self::from,
        )
    }

    fn round_to_integral(self, round: Round) -> StatusAnd<Self> {
        Fallback::from(self).round_to_integral(round).map(
            Self::from,
        )
    }

    fn next_up(self) -> StatusAnd<Self> {
        Fallback::from(self).next_up().map(Self::from)
    }

    fn from_bits(input: u128) -> Self {
        let (a, b) = (input, input >> F::BITS);
        DoubleFloat(
            F::from_bits(a & ((1 << F::BITS) - 1)),
            F::from_bits(b & ((1 << F::BITS) - 1)),
        )
    }

    fn from_u128_r(input: u128, round: Round) -> StatusAnd<Self> {
        Fallback::from_u128_r(input, round).map(Self::from)
    }

    fn from_str_r(s: &str, round: Round) -> Result<StatusAnd<Self>, ParseError> {
        Fallback::from_str_r(s, round).map(|r| r.map(Self::from))
    }

    fn to_bits(self) -> u128 {
        self.0.to_bits() | (self.1.to_bits() << F::BITS)
    }

    fn to_u128_r(self, width: usize, round: Round, is_exact: &mut bool) -> StatusAnd<u128> {
        Fallback::from(self).to_u128_r(width, round, is_exact)
    }

    fn cmp_abs_normal(self, rhs: Self) -> Ordering {
        self.0.cmp_abs_normal(rhs.0).then_with(|| {
            let result = self.1.cmp_abs_normal(rhs.1);
            if result != Ordering::Equal {
                let against = self.0.is_negative() ^ self.1.is_negative();
                let rhs_against = rhs.0.is_negative() ^ rhs.1.is_negative();
                (!against).cmp(&!rhs_against).then_with(|| if against {
                    result.reverse()
                } else {
                    result
                })
            } else {
                result
            }
        })
    }

    fn bitwise_eq(self, rhs: Self) -> bool {
        self.0.bitwise_eq(rhs.0) && self.1.bitwise_eq(rhs.1)
    }

    fn is_negative(self) -> bool {
        self.0.is_negative()
    }

    fn is_denormal(self) -> bool {
        self.category() == Category::Normal &&
            (self.0.is_denormal() || self.0.is_denormal() ||
          // (double)(Hi + Lo) == Hi defines a normal number.
          !(self.0 + self.1).value.bitwise_eq(self.0))
    }

    fn is_signaling(self) -> bool {
        self.0.is_signaling()
    }

    fn category(self) -> Category {
        self.0.category()
    }

    fn get_exact_inverse(self) -> Option<Self> {
        Fallback::from(self).get_exact_inverse().map(Self::from)
    }

    fn ilogb(self) -> ExpInt {
        self.0.ilogb()
    }

    fn scalbn_r(self, exp: ExpInt, round: Round) -> Self {
        DoubleFloat(self.0.scalbn_r(exp, round), self.1.scalbn_r(exp, round))
    }

    fn frexp_r(self, exp: &mut ExpInt, round: Round) -> Self {
        let a = self.0.frexp_r(exp, round);
        let mut b = self.1;
        if self.category() == Category::Normal {
            b = b.scalbn_r(-*exp, round);
        }
        DoubleFloat(a, b)
    }
}
