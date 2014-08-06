// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A Big integer (signed version: `BigInt`, unsigned version: `BigUint`).
//!
//! A `BigUint` is represented as an array of `BigDigit`s.
//! A `BigInt` is a combination of `BigUint` and `Sign`.
//!
//! Common numerical operations are overloaded, so we can treat them
//! the same way we treat other numbers.
//!
//! ## Example
//!
//! ```rust
//! use num::bigint::BigUint;
//! use std::num::{Zero, One};
//! use std::mem::replace;
//!
//! // Calculate large fibonacci numbers.
//! fn fib(n: uint) -> BigUint {
//!     let mut f0: BigUint = Zero::zero();
//!     let mut f1: BigUint = One::one();
//!     for _ in range(0, n) {
//!         let f2 = f0 + f1;
//!         // This is a low cost way of swapping f0 with f1 and f1 with f2.
//!         f0 = replace(&mut f1, f2);
//!     }
//!     f0
//! }
//!
//! // This is a very large number.
//! println!("fib(1000) = {}", fib(1000));
//! ```
//!
//! It's easy to generate large random numbers:
//!
//! ```rust
//! use num::bigint::{ToBigInt, RandBigInt};
//! use std::rand;
//!
//! let mut rng = rand::task_rng();
//! let a = rng.gen_bigint(1000u);
//!
//! let low = -10000i.to_bigint().unwrap();
//! let high = 10000i.to_bigint().unwrap();
//! let b = rng.gen_bigint_range(&low, &high);
//!
//! // Probably an even larger number.
//! println!("{}", a * b);
//! ```

use Integer;
use rand::Rng;

use std::{cmp, fmt};
use std::default::Default;
use std::from_str::FromStr;
use std::num::CheckedDiv;
use std::num::{ToPrimitive, FromPrimitive};
use std::num::{Zero, One, ToStrRadix, FromStrRadix};
use std::string::String;
use std::{uint, i64, u64};

/// A `BigDigit` is a `BigUint`'s composing element.
pub type BigDigit = u32;

/// A `DoubleBigDigit` is the internal type used to do the computations.  Its
/// size is the double of the size of `BigDigit`.
pub type DoubleBigDigit = u64;

pub static ZERO_BIG_DIGIT: BigDigit = 0;
static ZERO_VEC: [BigDigit, ..1] = [ZERO_BIG_DIGIT];

pub mod BigDigit {
    use super::BigDigit;
    use super::DoubleBigDigit;

    // `DoubleBigDigit` size dependent
    pub static bits: uint = 32;

    pub static base: DoubleBigDigit = 1 << bits;
    static lo_mask: DoubleBigDigit = (-1 as DoubleBigDigit) >> bits;

    #[inline]
    fn get_hi(n: DoubleBigDigit) -> BigDigit { (n >> bits) as BigDigit }
    #[inline]
    fn get_lo(n: DoubleBigDigit) -> BigDigit { (n & lo_mask) as BigDigit }

    /// Split one `DoubleBigDigit` into two `BigDigit`s.
    #[inline]
    pub fn from_doublebigdigit(n: DoubleBigDigit) -> (BigDigit, BigDigit) {
        (get_hi(n), get_lo(n))
    }

    /// Join two `BigDigit`s into one `DoubleBigDigit`
    #[inline]
    pub fn to_doublebigdigit(hi: BigDigit, lo: BigDigit) -> DoubleBigDigit {
        (lo as DoubleBigDigit) | ((hi as DoubleBigDigit) << bits)
    }
}

/// A big unsigned integer type.
///
/// A `BigUint`-typed value `BigUint { data: vec!(a, b, c) }` represents a number
/// `(a + b * BigDigit::base + c * BigDigit::base^2)`.
#[deriving(Clone)]
pub struct BigUint {
    data: Vec<BigDigit>
}

impl PartialEq for BigUint {
    #[inline]
    fn eq(&self, other: &BigUint) -> bool {
        match self.cmp(other) { Equal => true, _ => false }
    }
}
impl Eq for BigUint {}

impl PartialOrd for BigUint {
    #[inline]
    fn partial_cmp(&self, other: &BigUint) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigUint {
    #[inline]
    fn cmp(&self, other: &BigUint) -> Ordering {
        let (s_len, o_len) = (self.data.len(), other.data.len());
        if s_len < o_len { return Less; }
        if s_len > o_len { return Greater;  }

        for (&self_i, &other_i) in self.data.iter().rev().zip(other.data.iter().rev()) {
            if self_i < other_i { return Less; }
            if self_i > other_i { return Greater; }
        }
        return Equal;
    }
}

impl Default for BigUint {
    #[inline]
    fn default() -> BigUint { Zero::zero() }
}

impl fmt::Show for BigUint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_str_radix(10))
    }
}

impl FromStr for BigUint {
    #[inline]
    fn from_str(s: &str) -> Option<BigUint> {
        FromStrRadix::from_str_radix(s, 10)
    }
}

impl Num for BigUint {}

impl BitAnd<BigUint, BigUint> for BigUint {
    fn bitand(&self, other: &BigUint) -> BigUint {
        BigUint::new(self.data.iter().zip(other.data.iter()).map(|(ai, bi)| *ai & *bi).collect())
    }
}

impl BitOr<BigUint, BigUint> for BigUint {
    fn bitor(&self, other: &BigUint) -> BigUint {
        let zeros = ZERO_VEC.iter().cycle();
        let (a, b) = if self.data.len() > other.data.len() { (self, other) } else { (other, self) };
        let ored = a.data.iter().zip(b.data.iter().chain(zeros)).map(
            |(ai, bi)| *ai | *bi
        ).collect();
        return BigUint::new(ored);
    }
}

impl BitXor<BigUint, BigUint> for BigUint {
    fn bitxor(&self, other: &BigUint) -> BigUint {
        let zeros = ZERO_VEC.iter().cycle();
        let (a, b) = if self.data.len() > other.data.len() { (self, other) } else { (other, self) };
        let xored = a.data.iter().zip(b.data.iter().chain(zeros)).map(
            |(ai, bi)| *ai ^ *bi
        ).collect();
        return BigUint::new(xored);
    }
}

impl Shl<uint, BigUint> for BigUint {
    #[inline]
    fn shl(&self, rhs: &uint) -> BigUint {
        let n_unit = *rhs / BigDigit::bits;
        let n_bits = *rhs % BigDigit::bits;
        return self.shl_unit(n_unit).shl_bits(n_bits);
    }
}

impl Shr<uint, BigUint> for BigUint {
    #[inline]
    fn shr(&self, rhs: &uint) -> BigUint {
        let n_unit = *rhs / BigDigit::bits;
        let n_bits = *rhs % BigDigit::bits;
        return self.shr_unit(n_unit).shr_bits(n_bits);
    }
}

impl Zero for BigUint {
    #[inline]
    fn zero() -> BigUint { BigUint::new(Vec::new()) }

    #[inline]
    fn is_zero(&self) -> bool { self.data.is_empty() }
}

impl One for BigUint {
    #[inline]
    fn one() -> BigUint { BigUint::new(vec!(1)) }
}

impl Unsigned for BigUint {}

impl Add<BigUint, BigUint> for BigUint {
    fn add(&self, other: &BigUint) -> BigUint {
        let zeros = ZERO_VEC.iter().cycle();
        let (a, b) = if self.data.len() > other.data.len() { (self, other) } else { (other, self) };

        let mut carry = 0;
        let mut sum: Vec<BigDigit> =  a.data.iter().zip(b.data.iter().chain(zeros)).map(|(ai, bi)| {
            let (hi, lo) = BigDigit::from_doublebigdigit(
                (*ai as DoubleBigDigit) + (*bi as DoubleBigDigit) + (carry as DoubleBigDigit));
            carry = hi;
            lo
        }).collect();
        if carry != 0 { sum.push(carry); }
        return BigUint::new(sum);
    }
}

impl Sub<BigUint, BigUint> for BigUint {
    fn sub(&self, other: &BigUint) -> BigUint {
        let new_len = cmp::max(self.data.len(), other.data.len());
        let zeros = ZERO_VEC.iter().cycle();
        let (a, b) = (self.data.iter().chain(zeros.clone()), other.data.iter().chain(zeros));

        let mut borrow = 0i;
        let diff: Vec<BigDigit> =  a.take(new_len).zip(b).map(|(ai, bi)| {
            let (hi, lo) = BigDigit::from_doublebigdigit(
                BigDigit::base
                    + (*ai as DoubleBigDigit)
                    - (*bi as DoubleBigDigit)
                    - (borrow as DoubleBigDigit)
            );
            /*
            hi * (base) + lo == 1*(base) + ai - bi - borrow
            => ai - bi - borrow < 0 <=> hi == 0
            */
            borrow = if hi == 0 { 1 } else { 0 };
            lo
        }).collect();

        assert!(borrow == 0,
                "Cannot subtract other from self because other is larger than self.");
        return BigUint::new(diff);
    }
}

impl Mul<BigUint, BigUint> for BigUint {
    fn mul(&self, other: &BigUint) -> BigUint {
        if self.is_zero() || other.is_zero() { return Zero::zero(); }

        let (s_len, o_len) = (self.data.len(), other.data.len());
        if s_len == 1 { return mul_digit(other, self.data.as_slice()[0]);  }
        if o_len == 1 { return mul_digit(self,  other.data.as_slice()[0]); }

        // Using Karatsuba multiplication
        // (a1 * base + a0) * (b1 * base + b0)
        // = a1*b1 * base^2 +
        //   (a1*b1 + a0*b0 - (a1-b0)*(b1-a0)) * base +
        //   a0*b0
        let half_len = cmp::max(s_len, o_len) / 2;
        let (s_hi, s_lo) = cut_at(self,  half_len);
        let (o_hi, o_lo) = cut_at(other, half_len);

        let ll = s_lo * o_lo;
        let hh = s_hi * o_hi;
        let mm = {
            let (s1, n1) = sub_sign(s_hi, s_lo);
            let (s2, n2) = sub_sign(o_hi, o_lo);
            match (s1, s2) {
                (Equal, _) | (_, Equal) => hh + ll,
                (Less, Greater) | (Greater, Less) => hh + ll + (n1 * n2),
                (Less, Less) | (Greater, Greater) => hh + ll - (n1 * n2)
            }
        };

        return ll + mm.shl_unit(half_len) + hh.shl_unit(half_len * 2);


        fn mul_digit(a: &BigUint, n: BigDigit) -> BigUint {
            if n == 0 { return Zero::zero(); }
            if n == 1 { return (*a).clone(); }

            let mut carry = 0;
            let mut prod: Vec<BigDigit> = a.data.iter().map(|ai| {
                let (hi, lo) = BigDigit::from_doublebigdigit(
                    (*ai as DoubleBigDigit) * (n as DoubleBigDigit) + (carry as DoubleBigDigit)
                );
                carry = hi;
                lo
            }).collect();
            if carry != 0 { prod.push(carry); }
            return BigUint::new(prod);
        }

        #[inline]
        fn cut_at(a: &BigUint, n: uint) -> (BigUint, BigUint) {
            let mid = cmp::min(a.data.len(), n);
            return (BigUint::from_slice(a.data.slice(mid, a.data.len())),
                    BigUint::from_slice(a.data.slice(0, mid)));
        }

        #[inline]
        fn sub_sign(a: BigUint, b: BigUint) -> (Ordering, BigUint) {
            match a.cmp(&b) {
                Less    => (Less,    b - a),
                Greater => (Greater, a - b),
                _       => (Equal,   Zero::zero())
            }
        }
    }
}

impl Div<BigUint, BigUint> for BigUint {
    #[inline]
    fn div(&self, other: &BigUint) -> BigUint {
        let (q, _) = self.div_rem(other);
        return q;
    }
}

impl Rem<BigUint, BigUint> for BigUint {
    #[inline]
    fn rem(&self, other: &BigUint) -> BigUint {
        let (_, r) = self.div_rem(other);
        return r;
    }
}

impl Neg<BigUint> for BigUint {
    #[inline]
    fn neg(&self) -> BigUint { fail!() }
}

impl CheckedAdd for BigUint {
    #[inline]
    fn checked_add(&self, v: &BigUint) -> Option<BigUint> {
        return Some(self.add(v));
    }
}

impl CheckedSub for BigUint {
    #[inline]
    fn checked_sub(&self, v: &BigUint) -> Option<BigUint> {
        if *self < *v {
            return None;
        }
        return Some(self.sub(v));
    }
}

impl CheckedMul for BigUint {
    #[inline]
    fn checked_mul(&self, v: &BigUint) -> Option<BigUint> {
        return Some(self.mul(v));
    }
}

impl CheckedDiv for BigUint {
    #[inline]
    fn checked_div(&self, v: &BigUint) -> Option<BigUint> {
        if v.is_zero() {
            return None;
        }
        return Some(self.div(v));
    }
}

impl Integer for BigUint {
    #[inline]
    fn div_rem(&self, other: &BigUint) -> (BigUint, BigUint) {
        self.div_mod_floor(other)
    }

    #[inline]
    fn div_floor(&self, other: &BigUint) -> BigUint {
        let (d, _) = self.div_mod_floor(other);
        return d;
    }

    #[inline]
    fn mod_floor(&self, other: &BigUint) -> BigUint {
        let (_, m) = self.div_mod_floor(other);
        return m;
    }

    fn div_mod_floor(&self, other: &BigUint) -> (BigUint, BigUint) {
        if other.is_zero() { fail!() }
        if self.is_zero() { return (Zero::zero(), Zero::zero()); }
        if *other == One::one() { return ((*self).clone(), Zero::zero()); }

        match self.cmp(other) {
            Less    => return (Zero::zero(), (*self).clone()),
            Equal   => return (One::one(), Zero::zero()),
            Greater => {} // Do nothing
        }

        let mut shift = 0;
        let mut n = *other.data.last().unwrap();
        while n < (1 << BigDigit::bits - 2) {
            n <<= 1;
            shift += 1;
        }
        assert!(shift < BigDigit::bits);
        let (d, m) = div_mod_floor_inner(self << shift, other << shift);
        return (d, m >> shift);


        fn div_mod_floor_inner(a: BigUint, b: BigUint) -> (BigUint, BigUint) {
            let mut m = a;
            let mut d: BigUint = Zero::zero();
            let mut n = 1;
            while m >= b {
                let (d0, d_unit, b_unit) = div_estimate(&m, &b, n);
                let mut d0 = d0;
                let mut prod = b * d0;
                while prod > m {
                    // FIXME(#5992): assignment operator overloads
                    // d0 -= d_unit
                    d0   = d0 - d_unit;
                    // FIXME(#5992): assignment operator overloads
                    // prod -= b_unit;
                    prod = prod - b_unit
                }
                if d0.is_zero() {
                    n = 2;
                    continue;
                }
                n = 1;
                // FIXME(#5992): assignment operator overloads
                // d += d0;
                d = d + d0;
                // FIXME(#5992): assignment operator overloads
                // m -= prod;
                m = m - prod;
            }
            return (d, m);
        }


        fn div_estimate(a: &BigUint, b: &BigUint, n: uint)
            -> (BigUint, BigUint, BigUint) {
            if a.data.len() < n {
                return (Zero::zero(), Zero::zero(), (*a).clone());
            }

            let an = a.data.tailn(a.data.len() - n);
            let bn = *b.data.last().unwrap();
            let mut d = Vec::with_capacity(an.len());
            let mut carry = 0;
            for elt in an.iter().rev() {
                let ai = BigDigit::to_doublebigdigit(carry, *elt);
                let di = ai / (bn as DoubleBigDigit);
                assert!(di < BigDigit::base);
                carry = (ai % (bn as DoubleBigDigit)) as BigDigit;
                d.push(di as BigDigit)
            }
            d.reverse();

            let shift = (a.data.len() - an.len()) - (b.data.len() - 1);
            if shift == 0 {
                return (BigUint::new(d), One::one(), (*b).clone());
            }
            let one: BigUint = One::one();
            return (BigUint::new(d).shl_unit(shift),
                    one.shl_unit(shift),
                    b.shl_unit(shift));
        }
    }

    /// Calculates the Greatest Common Divisor (GCD) of the number and `other`.
    ///
    /// The result is always positive.
    #[inline]
    fn gcd(&self, other: &BigUint) -> BigUint {
        // Use Euclid's algorithm
        let mut m = (*self).clone();
        let mut n = (*other).clone();
        while !m.is_zero() {
            let temp = m;
            m = n % temp;
            n = temp;
        }
        return n;
    }

    /// Calculates the Lowest Common Multiple (LCM) of the number and `other`.
    #[inline]
    fn lcm(&self, other: &BigUint) -> BigUint { ((*self * *other) / self.gcd(other)) }

    /// Deprecated, use `is_multiple_of` instead.
    #[deprecated = "function renamed to `is_multiple_of`"]
    #[inline]
    fn divides(&self, other: &BigUint) -> bool { return self.is_multiple_of(other); }

    /// Returns `true` if the number is a multiple of `other`.
    #[inline]
    fn is_multiple_of(&self, other: &BigUint) -> bool { (*self % *other).is_zero() }

    /// Returns `true` if the number is divisible by `2`.
    #[inline]
    fn is_even(&self) -> bool {
        // Considering only the last digit.
        match self.data.as_slice().head() {
            Some(x) => x.is_even(),
            None => true
        }
    }

    /// Returns `true` if the number is not divisible by `2`.
    #[inline]
    fn is_odd(&self) -> bool { !self.is_even() }
}

impl ToPrimitive for BigUint {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.to_u64().and_then(|n| {
            // If top bit of u64 is set, it's too large to convert to i64.
            if n >> 63 == 0 {
                Some(n as i64)
            } else {
                None
            }
        })
    }

    // `DoubleBigDigit` size dependent
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        match self.data.len() {
            0 => Some(0),
            1 => Some(self.data.as_slice()[0] as u64),
            2 => Some(BigDigit::to_doublebigdigit(self.data.as_slice()[1], self.data.as_slice()[0])
                      as u64),
            _ => None
        }
    }
}

impl FromPrimitive for BigUint {
    #[inline]
    fn from_i64(n: i64) -> Option<BigUint> {
        if n > 0 {
            FromPrimitive::from_u64(n as u64)
        } else if n == 0 {
            Some(Zero::zero())
        } else {
            None
        }
    }

    // `DoubleBigDigit` size dependent
    #[inline]
    fn from_u64(n: u64) -> Option<BigUint> {
        let n = match BigDigit::from_doublebigdigit(n) {
            (0,  0)  => Zero::zero(),
            (0,  n0) => BigUint::new(vec!(n0)),
            (n1, n0) => BigUint::new(vec!(n0, n1))
        };
        Some(n)
    }
}

/// A generic trait for converting a value to a `BigUint`.
pub trait ToBigUint {
    /// Converts the value of `self` to a `BigUint`.
    fn to_biguint(&self) -> Option<BigUint>;
}

impl ToBigUint for BigInt {
    #[inline]
    fn to_biguint(&self) -> Option<BigUint> {
        if self.sign == Plus {
            Some(self.data.clone())
        } else if self.sign == Zero {
            Some(Zero::zero())
        } else {
            None
        }
    }
}

impl ToBigUint for BigUint {
    #[inline]
    fn to_biguint(&self) -> Option<BigUint> {
        Some(self.clone())
    }
}

macro_rules! impl_to_biguint(
    ($T:ty, $from_ty:path) => {
        impl ToBigUint for $T {
            #[inline]
            fn to_biguint(&self) -> Option<BigUint> {
                $from_ty(*self)
            }
        }
    }
)

impl_to_biguint!(int,  FromPrimitive::from_int)
impl_to_biguint!(i8,   FromPrimitive::from_i8)
impl_to_biguint!(i16,  FromPrimitive::from_i16)
impl_to_biguint!(i32,  FromPrimitive::from_i32)
impl_to_biguint!(i64,  FromPrimitive::from_i64)
impl_to_biguint!(uint, FromPrimitive::from_uint)
impl_to_biguint!(u8,   FromPrimitive::from_u8)
impl_to_biguint!(u16,  FromPrimitive::from_u16)
impl_to_biguint!(u32,  FromPrimitive::from_u32)
impl_to_biguint!(u64,  FromPrimitive::from_u64)

impl ToStrRadix for BigUint {
    fn to_str_radix(&self, radix: uint) -> String {
        assert!(1 < radix && radix <= 16, "The radix must be within (1, 16]");
        let (base, max_len) = get_radix_base(radix);
        if base == BigDigit::base {
            return fill_concat(self.data.as_slice(), radix, max_len)
        }
        return fill_concat(convert_base(self, base).as_slice(), radix, max_len);

        fn convert_base(n: &BigUint, base: DoubleBigDigit) -> Vec<BigDigit> {
            let divider    = base.to_biguint().unwrap();
            let mut result = Vec::new();
            let mut m      = n.clone();
            while m >= divider {
                let (d, m0) = m.div_mod_floor(&divider);
                result.push(m0.to_uint().unwrap() as BigDigit);
                m = d;
            }
            if !m.is_zero() {
                result.push(m.to_uint().unwrap() as BigDigit);
            }
            return result;
        }

        fn fill_concat(v: &[BigDigit], radix: uint, l: uint) -> String {
            if v.is_empty() {
                return "0".to_string()
            }
            let mut s = String::with_capacity(v.len() * l);
            for n in v.iter().rev() {
                let ss = (*n as uint).to_str_radix(radix);
                s.push_str("0".repeat(l - ss.len()).as_slice());
                s.push_str(ss.as_slice());
            }
            s.as_slice().trim_left_chars('0').to_string()
        }
    }
}

impl FromStrRadix for BigUint {
    /// Creates and initializes a `BigUint`.
    #[inline]
    fn from_str_radix(s: &str, radix: uint) -> Option<BigUint> {
        BigUint::parse_bytes(s.as_bytes(), radix)
    }
}

impl BigUint {
    /// Creates and initializes a `BigUint`.
    ///
    /// The digits are be in base 2^32.
    #[inline]
    pub fn new(mut digits: Vec<BigDigit>) -> BigUint {
        // omit trailing zeros
        let new_len = digits.iter().rposition(|n| *n != 0).map_or(0, |p| p + 1);
        digits.truncate(new_len);
        BigUint { data: digits }
    }

    /// Creates and initializes a `BigUint`.
    ///
    /// The digits are be in base 2^32.
    #[inline]
    pub fn from_slice(slice: &[BigDigit]) -> BigUint {
        BigUint::new(Vec::from_slice(slice))
    }

    /// Creates and initializes a `BigUint`.
    pub fn parse_bytes(buf: &[u8], radix: uint) -> Option<BigUint> {
        let (base, unit_len) = get_radix_base(radix);
        let base_num = match base.to_biguint() {
            Some(base_num) => base_num,
            None => { return None; }
        };

        let mut end             = buf.len();
        let mut n: BigUint      = Zero::zero();
        let mut power: BigUint  = One::one();
        loop {
            let start = cmp::max(end, unit_len) - unit_len;
            match uint::parse_bytes(buf.slice(start, end), radix) {
                Some(d) => {
                    let d: Option<BigUint> = FromPrimitive::from_uint(d);
                    match d {
                        Some(d) => {
                            // FIXME(#5992): assignment operator overloads
                            // n += d * power;
                            n = n + d * power;
                        }
                        None => { return None; }
                    }
                }
                None => { return None; }
            }
            if end <= unit_len {
                return Some(n);
            }
            end -= unit_len;
            // FIXME(#5992): assignment operator overloads
            // power *= base_num;
            power = power * base_num;
        }
    }

    #[inline]
    fn shl_unit(&self, n_unit: uint) -> BigUint {
        if n_unit == 0 || self.is_zero() { return (*self).clone(); }

        BigUint::new(Vec::from_elem(n_unit, ZERO_BIG_DIGIT).append(self.data.as_slice()))
    }

    #[inline]
    fn shl_bits(&self, n_bits: uint) -> BigUint {
        if n_bits == 0 || self.is_zero() { return (*self).clone(); }

        let mut carry = 0;
        let mut shifted: Vec<BigDigit> = self.data.iter().map(|elem| {
            let (hi, lo) = BigDigit::from_doublebigdigit(
                (*elem as DoubleBigDigit) << n_bits | (carry as DoubleBigDigit)
            );
            carry = hi;
            lo
        }).collect();
        if carry != 0 { shifted.push(carry); }
        return BigUint::new(shifted);
    }

    #[inline]
    fn shr_unit(&self, n_unit: uint) -> BigUint {
        if n_unit == 0 { return (*self).clone(); }
        if self.data.len() < n_unit { return Zero::zero(); }
        return BigUint::from_slice(
            self.data.slice(n_unit, self.data.len())
        );
    }

    #[inline]
    fn shr_bits(&self, n_bits: uint) -> BigUint {
        if n_bits == 0 || self.data.is_empty() { return (*self).clone(); }

        let mut borrow = 0;
        let mut shifted_rev = Vec::with_capacity(self.data.len());
        for elem in self.data.iter().rev() {
            shifted_rev.push((*elem >> n_bits) | borrow);
            borrow = *elem << (BigDigit::bits - n_bits);
        }
        let shifted = { shifted_rev.reverse(); shifted_rev };
        return BigUint::new(shifted);
    }

    /// Determines the fewest bits necessary to express the `BigUint`.
    pub fn bits(&self) -> uint {
        if self.is_zero() { return 0; }
        let zeros = self.data.last().unwrap().leading_zeros();
        return self.data.len()*BigDigit::bits - (zeros as uint);
    }
}

// `DoubleBigDigit` size dependent
#[inline]
fn get_radix_base(radix: uint) -> (DoubleBigDigit, uint) {
    match radix {
        2  => (4294967296, 32),
        3  => (3486784401, 20),
        4  => (4294967296, 16),
        5  => (1220703125, 13),
        6  => (2176782336, 12),
        7  => (1977326743, 11),
        8  => (1073741824, 10),
        9  => (3486784401, 10),
        10 => (1000000000, 9),
        11 => (2357947691, 9),
        12 => (429981696,  8),
        13 => (815730721,  8),
        14 => (1475789056, 8),
        15 => (2562890625, 8),
        16 => (4294967296, 8),
        _  => fail!("The radix must be within (1, 16]")
    }
}

/// A Sign is a `BigInt`'s composing element.
#[deriving(PartialEq, PartialOrd, Eq, Ord, Clone, Show)]
pub enum Sign { Minus, Zero, Plus }

impl Neg<Sign> for Sign {
    /// Negate Sign value.
    #[inline]
    fn neg(&self) -> Sign {
        match *self {
          Minus => Plus,
          Zero  => Zero,
          Plus  => Minus
        }
    }
}

/// A big signed integer type.
#[deriving(Clone)]
pub struct BigInt {
    sign: Sign,
    data: BigUint
}

impl PartialEq for BigInt {
    #[inline]
    fn eq(&self, other: &BigInt) -> bool {
        self.cmp(other) == Equal
    }
}

impl Eq for BigInt {}

impl PartialOrd for BigInt {
    #[inline]
    fn partial_cmp(&self, other: &BigInt) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigInt {
    #[inline]
    fn cmp(&self, other: &BigInt) -> Ordering {
        let scmp = self.sign.cmp(&other.sign);
        if scmp != Equal { return scmp; }

        match self.sign {
            Zero  => Equal,
            Plus  => self.data.cmp(&other.data),
            Minus => other.data.cmp(&self.data),
        }
    }
}

impl Default for BigInt {
    #[inline]
    fn default() -> BigInt { Zero::zero() }
}

impl fmt::Show for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_str_radix(10))
    }
}

impl FromStr for BigInt {
    #[inline]
    fn from_str(s: &str) -> Option<BigInt> {
        FromStrRadix::from_str_radix(s, 10)
    }
}

impl Num for BigInt {}

impl Shl<uint, BigInt> for BigInt {
    #[inline]
    fn shl(&self, rhs: &uint) -> BigInt {
        BigInt::from_biguint(self.sign, self.data << *rhs)
    }
}

impl Shr<uint, BigInt> for BigInt {
    #[inline]
    fn shr(&self, rhs: &uint) -> BigInt {
        BigInt::from_biguint(self.sign, self.data >> *rhs)
    }
}

impl Zero for BigInt {
    #[inline]
    fn zero() -> BigInt {
        BigInt::from_biguint(Zero, Zero::zero())
    }

    #[inline]
    fn is_zero(&self) -> bool { self.sign == Zero }
}

impl One for BigInt {
    #[inline]
    fn one() -> BigInt {
        BigInt::from_biguint(Plus, One::one())
    }
}

impl Signed for BigInt {
    #[inline]
    fn abs(&self) -> BigInt {
        match self.sign {
            Plus | Zero => self.clone(),
            Minus => BigInt::from_biguint(Plus, self.data.clone())
        }
    }

    #[inline]
    fn abs_sub(&self, other: &BigInt) -> BigInt {
        if *self <= *other { Zero::zero() } else { *self - *other }
    }

    #[inline]
    fn signum(&self) -> BigInt {
        match self.sign {
            Plus  => BigInt::from_biguint(Plus, One::one()),
            Minus => BigInt::from_biguint(Minus, One::one()),
            Zero  => Zero::zero(),
        }
    }

    #[inline]
    fn is_positive(&self) -> bool { self.sign == Plus }

    #[inline]
    fn is_negative(&self) -> bool { self.sign == Minus }
}

impl Add<BigInt, BigInt> for BigInt {
    #[inline]
    fn add(&self, other: &BigInt) -> BigInt {
        match (self.sign, other.sign) {
            (Zero, _)      => other.clone(),
            (_,    Zero)   => self.clone(),
            (Plus, Plus)   => BigInt::from_biguint(Plus, self.data + other.data),
            (Plus, Minus)  => self - (-*other),
            (Minus, Plus)  => other - (-*self),
            (Minus, Minus) => -((-self) + (-*other))
        }
    }
}

impl Sub<BigInt, BigInt> for BigInt {
    #[inline]
    fn sub(&self, other: &BigInt) -> BigInt {
        match (self.sign, other.sign) {
            (Zero, _)    => -other,
            (_,    Zero) => self.clone(),
            (Plus, Plus) => match self.data.cmp(&other.data) {
                Less    => BigInt::from_biguint(Minus, other.data - self.data),
                Greater => BigInt::from_biguint(Plus, self.data - other.data),
                Equal   => Zero::zero()
            },
            (Plus, Minus) => self + (-*other),
            (Minus, Plus) => -((-self) + *other),
            (Minus, Minus) => (-other) - (-*self)
        }
    }
}

impl Mul<BigInt, BigInt> for BigInt {
    #[inline]
    fn mul(&self, other: &BigInt) -> BigInt {
        match (self.sign, other.sign) {
            (Zero, _)     | (_,     Zero)  => Zero::zero(),
            (Plus, Plus)  | (Minus, Minus) => {
                BigInt::from_biguint(Plus, self.data * other.data)
            },
            (Plus, Minus) | (Minus, Plus) => {
                BigInt::from_biguint(Minus, self.data * other.data)
            }
        }
    }
}

impl Div<BigInt, BigInt> for BigInt {
    #[inline]
    fn div(&self, other: &BigInt) -> BigInt {
        let (q, _) = self.div_rem(other);
        q
    }
}

impl Rem<BigInt, BigInt> for BigInt {
    #[inline]
    fn rem(&self, other: &BigInt) -> BigInt {
        let (_, r) = self.div_rem(other);
        r
    }
}

impl Neg<BigInt> for BigInt {
    #[inline]
    fn neg(&self) -> BigInt {
        BigInt::from_biguint(self.sign.neg(), self.data.clone())
    }
}

impl CheckedAdd for BigInt {
    #[inline]
    fn checked_add(&self, v: &BigInt) -> Option<BigInt> {
        return Some(self.add(v));
    }
}

impl CheckedSub for BigInt {
    #[inline]
    fn checked_sub(&self, v: &BigInt) -> Option<BigInt> {
        return Some(self.sub(v));
    }
}

impl CheckedMul for BigInt {
    #[inline]
    fn checked_mul(&self, v: &BigInt) -> Option<BigInt> {
        return Some(self.mul(v));
    }
}

impl CheckedDiv for BigInt {
    #[inline]
    fn checked_div(&self, v: &BigInt) -> Option<BigInt> {
        if v.is_zero() {
            return None;
        }
        return Some(self.div(v));
    }
}


impl Integer for BigInt {
    #[inline]
    fn div_rem(&self, other: &BigInt) -> (BigInt, BigInt) {
        // r.sign == self.sign
        let (d_ui, r_ui) = self.data.div_mod_floor(&other.data);
        let d = BigInt::from_biguint(Plus, d_ui);
        let r = BigInt::from_biguint(Plus, r_ui);
        match (self.sign, other.sign) {
            (_,    Zero)   => fail!(),
            (Plus, Plus)  | (Zero, Plus)  => ( d,  r),
            (Plus, Minus) | (Zero, Minus) => (-d,  r),
            (Minus, Plus)                 => (-d, -r),
            (Minus, Minus)                => ( d, -r)
        }
    }

    #[inline]
    fn div_floor(&self, other: &BigInt) -> BigInt {
        let (d, _) = self.div_mod_floor(other);
        d
    }

    #[inline]
    fn mod_floor(&self, other: &BigInt) -> BigInt {
        let (_, m) = self.div_mod_floor(other);
        m
    }

    fn div_mod_floor(&self, other: &BigInt) -> (BigInt, BigInt) {
        // m.sign == other.sign
        let (d_ui, m_ui) = self.data.div_rem(&other.data);
        let d = BigInt::from_biguint(Plus, d_ui);
        let m = BigInt::from_biguint(Plus, m_ui);
        match (self.sign, other.sign) {
            (_,    Zero)   => fail!(),
            (Plus, Plus)  | (Zero, Plus)  => (d, m),
            (Plus, Minus) | (Zero, Minus) => if m.is_zero() {
                (-d, Zero::zero())
            } else {
                (-d - One::one(), m + *other)
            },
            (Minus, Plus) => if m.is_zero() {
                (-d, Zero::zero())
            } else {
                (-d - One::one(), other - m)
            },
            (Minus, Minus) => (d, -m)
        }
    }

    /// Calculates the Greatest Common Divisor (GCD) of the number and `other`.
    ///
    /// The result is always positive.
    #[inline]
    fn gcd(&self, other: &BigInt) -> BigInt {
        BigInt::from_biguint(Plus, self.data.gcd(&other.data))
    }

    /// Calculates the Lowest Common Multiple (LCM) of the number and `other`.
    #[inline]
    fn lcm(&self, other: &BigInt) -> BigInt {
        BigInt::from_biguint(Plus, self.data.lcm(&other.data))
    }

    /// Deprecated, use `is_multiple_of` instead.
    #[deprecated = "function renamed to `is_multiple_of`"]
    #[inline]
    fn divides(&self, other: &BigInt) -> bool { return self.is_multiple_of(other); }

    /// Returns `true` if the number is a multiple of `other`.
    #[inline]
    fn is_multiple_of(&self, other: &BigInt) -> bool { self.data.is_multiple_of(&other.data) }

    /// Returns `true` if the number is divisible by `2`.
    #[inline]
    fn is_even(&self) -> bool { self.data.is_even() }

    /// Returns `true` if the number is not divisible by `2`.
    #[inline]
    fn is_odd(&self) -> bool { self.data.is_odd() }
}

impl ToPrimitive for BigInt {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        match self.sign {
            Plus  => self.data.to_i64(),
            Zero  => Some(0),
            Minus => {
                self.data.to_u64().and_then(|n| {
                    let m: u64 = 1 << 63;
                    if n < m {
                        Some(-(n as i64))
                    } else if n == m {
                        Some(i64::MIN)
                    } else {
                        None
                    }
                })
            }
        }
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        match self.sign {
            Plus => self.data.to_u64(),
            Zero => Some(0),
            Minus => None
        }
    }
}

impl FromPrimitive for BigInt {
    #[inline]
    fn from_i64(n: i64) -> Option<BigInt> {
        if n > 0 {
            FromPrimitive::from_u64(n as u64).and_then(|n| {
                Some(BigInt::from_biguint(Plus, n))
            })
        } else if n < 0 {
            FromPrimitive::from_u64(u64::MAX - (n as u64) + 1).and_then(
                |n| {
                    Some(BigInt::from_biguint(Minus, n))
                })
        } else {
            Some(Zero::zero())
        }
    }

    #[inline]
    fn from_u64(n: u64) -> Option<BigInt> {
        if n == 0 {
            Some(Zero::zero())
        } else {
            FromPrimitive::from_u64(n).and_then(|n| {
                Some(BigInt::from_biguint(Plus, n))
            })
        }
    }
}

/// A generic trait for converting a value to a `BigInt`.
pub trait ToBigInt {
    /// Converts the value of `self` to a `BigInt`.
    fn to_bigint(&self) -> Option<BigInt>;
}

impl ToBigInt for BigInt {
    #[inline]
    fn to_bigint(&self) -> Option<BigInt> {
        Some(self.clone())
    }
}

impl ToBigInt for BigUint {
    #[inline]
    fn to_bigint(&self) -> Option<BigInt> {
        if self.is_zero() {
            Some(Zero::zero())
        } else {
            Some(BigInt { sign: Plus, data: self.clone() })
        }
    }
}

macro_rules! impl_to_bigint(
    ($T:ty, $from_ty:path) => {
        impl ToBigInt for $T {
            #[inline]
            fn to_bigint(&self) -> Option<BigInt> {
                $from_ty(*self)
            }
        }
    }
)

impl_to_bigint!(int,  FromPrimitive::from_int)
impl_to_bigint!(i8,   FromPrimitive::from_i8)
impl_to_bigint!(i16,  FromPrimitive::from_i16)
impl_to_bigint!(i32,  FromPrimitive::from_i32)
impl_to_bigint!(i64,  FromPrimitive::from_i64)
impl_to_bigint!(uint, FromPrimitive::from_uint)
impl_to_bigint!(u8,   FromPrimitive::from_u8)
impl_to_bigint!(u16,  FromPrimitive::from_u16)
impl_to_bigint!(u32,  FromPrimitive::from_u32)
impl_to_bigint!(u64,  FromPrimitive::from_u64)

impl ToStrRadix for BigInt {
    #[inline]
    fn to_str_radix(&self, radix: uint) -> String {
        match self.sign {
            Plus  => self.data.to_str_radix(radix),
            Zero  => "0".to_string(),
            Minus => format!("-{}", self.data.to_str_radix(radix)),
        }
    }
}

impl FromStrRadix for BigInt {
    /// Creates and initializes a BigInt.
    #[inline]
    fn from_str_radix(s: &str, radix: uint) -> Option<BigInt> {
        BigInt::parse_bytes(s.as_bytes(), radix)
    }
}

pub trait RandBigInt {
    /// Generate a random `BigUint` of the given bit size.
    fn gen_biguint(&mut self, bit_size: uint) -> BigUint;

    /// Generate a random BigInt of the given bit size.
    fn gen_bigint(&mut self, bit_size: uint) -> BigInt;

    /// Generate a random `BigUint` less than the given bound. Fails
    /// when the bound is zero.
    fn gen_biguint_below(&mut self, bound: &BigUint) -> BigUint;

    /// Generate a random `BigUint` within the given range. The lower
    /// bound is inclusive; the upper bound is exclusive. Fails when
    /// the upper bound is not greater than the lower bound.
    fn gen_biguint_range(&mut self, lbound: &BigUint, ubound: &BigUint) -> BigUint;

    /// Generate a random `BigInt` within the given range. The lower
    /// bound is inclusive; the upper bound is exclusive. Fails when
    /// the upper bound is not greater than the lower bound.
    fn gen_bigint_range(&mut self, lbound: &BigInt, ubound: &BigInt) -> BigInt;
}

impl<R: Rng> RandBigInt for R {
    fn gen_biguint(&mut self, bit_size: uint) -> BigUint {
        let (digits, rem) = bit_size.div_rem(&BigDigit::bits);
        let mut data = Vec::with_capacity(digits+1);
        for _ in range(0, digits) {
            data.push(self.gen());
        }
        if rem > 0 {
            let final_digit: BigDigit = self.gen();
            data.push(final_digit >> (BigDigit::bits - rem));
        }
        BigUint::new(data)
    }

    fn gen_bigint(&mut self, bit_size: uint) -> BigInt {
        // Generate a random BigUint...
        let biguint = self.gen_biguint(bit_size);
        // ...and then randomly assign it a Sign...
        let sign = if biguint.is_zero() {
            // ...except that if the BigUint is zero, we need to try
            // again with probability 0.5. This is because otherwise,
            // the probability of generating a zero BigInt would be
            // double that of any other number.
            if self.gen() {
                return self.gen_bigint(bit_size);
            } else {
                Zero
            }
        } else if self.gen() {
            Plus
        } else {
            Minus
        };
        BigInt::from_biguint(sign, biguint)
    }

    fn gen_biguint_below(&mut self, bound: &BigUint) -> BigUint {
        assert!(!bound.is_zero());
        let bits = bound.bits();
        loop {
            let n = self.gen_biguint(bits);
            if n < *bound { return n; }
        }
    }

    fn gen_biguint_range(&mut self,
                         lbound: &BigUint,
                         ubound: &BigUint)
                         -> BigUint {
        assert!(*lbound < *ubound);
        return *lbound + self.gen_biguint_below(&(*ubound - *lbound));
    }

    fn gen_bigint_range(&mut self,
                        lbound: &BigInt,
                        ubound: &BigInt)
                        -> BigInt {
        assert!(*lbound < *ubound);
        let delta = (*ubound - *lbound).to_biguint().unwrap();
        return *lbound + self.gen_biguint_below(&delta).to_bigint().unwrap();
    }
}

impl BigInt {
    /// Creates and initializes a BigInt.
    ///
    /// The digits are be in base 2^32.
    #[inline]
    pub fn new(sign: Sign, digits: Vec<BigDigit>) -> BigInt {
        BigInt::from_biguint(sign, BigUint::new(digits))
    }

    /// Creates and initializes a `BigInt`.
    ///
    /// The digits are be in base 2^32.
    #[inline]
    pub fn from_biguint(sign: Sign, data: BigUint) -> BigInt {
        if sign == Zero || data.is_zero() {
            return BigInt { sign: Zero, data: Zero::zero() };
        }
        BigInt { sign: sign, data: data }
    }

    /// Creates and initializes a `BigInt`.
    #[inline]
    pub fn from_slice(sign: Sign, slice: &[BigDigit]) -> BigInt {
        BigInt::from_biguint(sign, BigUint::from_slice(slice))
    }

    /// Creates and initializes a `BigInt`.
    pub fn parse_bytes(buf: &[u8], radix: uint) -> Option<BigInt> {
        if buf.is_empty() { return None; }
        let mut sign  = Plus;
        let mut start = 0;
        if buf[0] == b'-' {
            sign  = Minus;
            start = 1;
        }
        return BigUint::parse_bytes(buf.slice(start, buf.len()), radix)
            .map(|bu| BigInt::from_biguint(sign, bu));
    }

    /// Converts this `BigInt` into a `BigUint`, if it's not negative.
    #[inline]
    pub fn to_biguint(&self) -> Option<BigUint> {
        match self.sign {
            Plus => Some(self.data.clone()),
            Zero => Some(Zero::zero()),
            Minus => None
        }
    }
}

#[cfg(test)]
mod biguint_tests {
    use Integer;
    use super::{BigDigit, BigUint, ToBigUint};
    use super::{Plus, BigInt, RandBigInt, ToBigInt};

    use std::cmp::{Less, Equal, Greater};
    use std::from_str::FromStr;
    use std::i64;
    use std::num::{Zero, One, FromStrRadix, ToStrRadix};
    use std::num::{ToPrimitive, FromPrimitive};
    use std::num::CheckedDiv;
    use std::rand::task_rng;
    use std::u64;

    #[test]
    fn test_from_slice() {
        fn check(slice: &[BigDigit], data: &[BigDigit]) {
            assert!(data == BigUint::from_slice(slice).data.as_slice());
        }
        check([1], [1]);
        check([0, 0, 0], []);
        check([1, 2, 0, 0], [1, 2]);
        check([0, 0, 1, 2], [0, 0, 1, 2]);
        check([0, 0, 1, 2, 0, 0], [0, 0, 1, 2]);
        check([-1], [-1]);
    }

    #[test]
    fn test_cmp() {
        let data: Vec<BigUint> = [ &[], &[1], &[2], &[-1], &[0, 1], &[2, 1], &[1, 1, 1]  ]
            .iter().map(|v| BigUint::from_slice(*v)).collect();
        for (i, ni) in data.iter().enumerate() {
            for (j0, nj) in data.slice(i, data.len()).iter().enumerate() {
                let j = j0 + i;
                if i == j {
                    assert_eq!(ni.cmp(nj), Equal);
                    assert_eq!(nj.cmp(ni), Equal);
                    assert_eq!(ni, nj);
                    assert!(!(ni != nj));
                    assert!(ni <= nj);
                    assert!(ni >= nj);
                    assert!(!(ni < nj));
                    assert!(!(ni > nj));
                } else {
                    assert_eq!(ni.cmp(nj), Less);
                    assert_eq!(nj.cmp(ni), Greater);

                    assert!(!(ni == nj));
                    assert!(ni != nj);

                    assert!(ni <= nj);
                    assert!(!(ni >= nj));
                    assert!(ni < nj);
                    assert!(!(ni > nj));

                    assert!(!(nj <= ni));
                    assert!(nj >= ni);
                    assert!(!(nj < ni));
                    assert!(nj > ni);
                }
            }
        }
    }

    #[test]
    fn test_bitand() {
        fn check(left: &[BigDigit],
                 right: &[BigDigit],
                 expected: &[BigDigit]) {
            assert_eq!(BigUint::from_slice(left) & BigUint::from_slice(right),
                       BigUint::from_slice(expected));
        }
        check([], [], []);
        check([268, 482, 17],
              [964, 54],
              [260, 34]);
    }

    #[test]
    fn test_bitor() {
        fn check(left: &[BigDigit],
                 right: &[BigDigit],
                 expected: &[BigDigit]) {
            assert_eq!(BigUint::from_slice(left) | BigUint::from_slice(right),
                       BigUint::from_slice(expected));
        }
        check([], [], []);
        check([268, 482, 17],
              [964, 54],
              [972, 502, 17]);
    }

    #[test]
    fn test_bitxor() {
        fn check(left: &[BigDigit],
                 right: &[BigDigit],
                 expected: &[BigDigit]) {
            assert_eq!(BigUint::from_slice(left) ^ BigUint::from_slice(right),
                       BigUint::from_slice(expected));
        }
        check([], [], []);
        check([268, 482, 17],
              [964, 54],
              [712, 468, 17]);
    }

    #[test]
    fn test_shl() {
        fn check(s: &str, shift: uint, ans: &str) {
            let opt_biguint: Option<BigUint> = FromStrRadix::from_str_radix(s, 16);
            let bu = (opt_biguint.unwrap() << shift).to_str_radix(16);
            assert_eq!(bu.as_slice(), ans);
        }

        check("0", 3, "0");
        check("1", 3, "8");

        check("1\
               0000\
               0000\
               0000\
               0001\
               0000\
               0000\
               0000\
               0001",
              3,
              "8\
               0000\
               0000\
               0000\
               0008\
               0000\
               0000\
               0000\
               0008");
        check("1\
               0000\
               0001\
               0000\
               0001",
              2,
              "4\
               0000\
               0004\
               0000\
               0004");
        check("1\
               0001\
               0001",
              1,
              "2\
               0002\
               0002");

        check("\
              4000\
              0000\
              0000\
              0000",
              3,
              "2\
              0000\
              0000\
              0000\
              0000");
        check("4000\
              0000",
              2,
              "1\
              0000\
              0000");
        check("4000",
              2,
              "1\
              0000");

        check("4000\
              0000\
              0000\
              0000",
              67,
              "2\
              0000\
              0000\
              0000\
              0000\
              0000\
              0000\
              0000\
              0000");
        check("4000\
              0000",
              35,
              "2\
              0000\
              0000\
              0000\
              0000");
        check("4000",
              19,
              "2\
              0000\
              0000");

        check("fedc\
              ba98\
              7654\
              3210\
              fedc\
              ba98\
              7654\
              3210",
              4,
              "f\
              edcb\
              a987\
              6543\
              210f\
              edcb\
              a987\
              6543\
              2100");
        check("88887777666655554444333322221111", 16,
              "888877776666555544443333222211110000");
    }

    #[test]
    fn test_shr() {
        fn check(s: &str, shift: uint, ans: &str) {
            let opt_biguint: Option<BigUint> =
                FromStrRadix::from_str_radix(s, 16);
            let bu = (opt_biguint.unwrap() >> shift).to_str_radix(16);
            assert_eq!(bu.as_slice(), ans);
        }

        check("0", 3, "0");
        check("f", 3, "1");

        check("1\
              0000\
              0000\
              0000\
              0001\
              0000\
              0000\
              0000\
              0001",
              3,
              "2000\
              0000\
              0000\
              0000\
              2000\
              0000\
              0000\
              0000");
        check("1\
              0000\
              0001\
              0000\
              0001",
              2,
              "4000\
              0000\
              4000\
              0000");
        check("1\
              0001\
              0001",
              1,
              "8000\
              8000");

        check("2\
              0000\
              0000\
              0000\
              0001\
              0000\
              0000\
              0000\
              0001",
              67,
              "4000\
              0000\
              0000\
              0000");
        check("2\
              0000\
              0001\
              0000\
              0001",
              35,
              "4000\
              0000");
        check("2\
              0001\
              0001",
              19,
              "4000");

        check("1\
              0000\
              0000\
              0000\
              0000",
              1,
              "8000\
              0000\
              0000\
              0000");
        check("1\
              0000\
              0000",
              1,
              "8000\
              0000");
        check("1\
              0000",
              1,
              "8000");
        check("f\
              edcb\
              a987\
              6543\
              210f\
              edcb\
              a987\
              6543\
              2100",
              4,
              "fedc\
              ba98\
              7654\
              3210\
              fedc\
              ba98\
              7654\
              3210");

        check("888877776666555544443333222211110000", 16,
              "88887777666655554444333322221111");
    }

    // `DoubleBigDigit` size dependent
    #[test]
    fn test_convert_i64() {
        fn check(b1: BigUint, i: i64) {
            let b2: BigUint = FromPrimitive::from_i64(i).unwrap();
            assert!(b1 == b2);
            assert!(b1.to_i64().unwrap() == i);
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);
        check(i64::MAX.to_biguint().unwrap(), i64::MAX);

        check(BigUint::new(vec!(           )), 0);
        check(BigUint::new(vec!( 1         )), (1 << (0*BigDigit::bits)));
        check(BigUint::new(vec!(-1         )), (1 << (1*BigDigit::bits)) - 1);
        check(BigUint::new(vec!( 0,  1     )), (1 << (1*BigDigit::bits)));
        check(BigUint::new(vec!(-1, -1 >> 1)), i64::MAX);

        assert_eq!(i64::MIN.to_biguint(), None);
        assert_eq!(BigUint::new(vec!(-1, -1    )).to_i64(), None);
        assert_eq!(BigUint::new(vec!( 0,  0,  1)).to_i64(), None);
        assert_eq!(BigUint::new(vec!(-1, -1, -1)).to_i64(), None);
    }

    // `DoubleBigDigit` size dependent
    #[test]
    fn test_convert_u64() {
        fn check(b1: BigUint, u: u64) {
            let b2: BigUint = FromPrimitive::from_u64(u).unwrap();
            assert!(b1 == b2);
            assert!(b1.to_u64().unwrap() == u);
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);
        check(u64::MIN.to_biguint().unwrap(), u64::MIN);
        check(u64::MAX.to_biguint().unwrap(), u64::MAX);

        check(BigUint::new(vec!(      )), 0);
        check(BigUint::new(vec!( 1    )), (1 << (0*BigDigit::bits)));
        check(BigUint::new(vec!(-1    )), (1 << (1*BigDigit::bits)) - 1);
        check(BigUint::new(vec!( 0,  1)), (1 << (1*BigDigit::bits)));
        check(BigUint::new(vec!(-1, -1)), u64::MAX);

        assert_eq!(BigUint::new(vec!( 0,  0,  1)).to_u64(), None);
        assert_eq!(BigUint::new(vec!(-1, -1, -1)).to_u64(), None);
    }

    #[test]
    fn test_convert_to_bigint() {
        fn check(n: BigUint, ans: BigInt) {
            assert_eq!(n.to_bigint().unwrap(), ans);
            assert_eq!(n.to_bigint().unwrap().to_biguint().unwrap(), n);
        }
        check(Zero::zero(), Zero::zero());
        check(BigUint::new(vec!(1,2,3)),
              BigInt::from_biguint(Plus, BigUint::new(vec!(1,2,3))));
    }

    static sum_triples: &'static [(&'static [BigDigit],
                                   &'static [BigDigit],
                                   &'static [BigDigit])] = &[
        (&[],          &[],       &[]),
        (&[],          &[ 1],     &[ 1]),
        (&[ 1],        &[ 1],     &[ 2]),
        (&[ 1],        &[ 1,  1], &[ 2,  1]),
        (&[ 1],        &[-1],     &[ 0,  1]),
        (&[ 1],        &[-1, -1], &[ 0,  0, 1]),
        (&[-1, -1],    &[-1, -1], &[-2, -1, 1]),
        (&[ 1,  1, 1], &[-1, -1], &[ 0,  1, 2]),
        (&[ 2,  2, 1], &[-1, -2], &[ 1,  1, 2])
    ];

    #[test]
    fn test_add() {
        for elm in sum_triples.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);

            assert!(a + b == c);
            assert!(b + a == c);
        }
    }

    #[test]
    fn test_sub() {
        for elm in sum_triples.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);

            assert!(c - a == b);
            assert!(c - b == a);
        }
    }

    #[test]
    #[should_fail]
    fn test_sub_fail_on_underflow() {
        let (a, b) : (BigUint, BigUint) = (Zero::zero(), One::one());
        a - b;
    }

    static mul_triples: &'static [(&'static [BigDigit],
                                   &'static [BigDigit],
                                   &'static [BigDigit])] = &[
        (&[],               &[],               &[]),
        (&[],               &[ 1],             &[]),
        (&[ 2],             &[],               &[]),
        (&[ 1],             &[ 1],             &[1]),
        (&[ 2],             &[ 3],             &[ 6]),
        (&[ 1],             &[ 1,  1,  1],     &[1, 1,  1]),
        (&[ 1,  2,  3],     &[ 3],             &[ 3,  6,  9]),
        (&[ 1,  1,  1],     &[-1],             &[-1, -1, -1]),
        (&[ 1,  2,  3],     &[-1],             &[-1, -2, -2, 2]),
        (&[ 1,  2,  3,  4], &[-1],             &[-1, -2, -2, -2, 3]),
        (&[-1],             &[-1],             &[ 1, -2]),
        (&[-1, -1],         &[-1],             &[ 1, -1, -2]),
        (&[-1, -1, -1],     &[-1],             &[ 1, -1, -1, -2]),
        (&[-1, -1, -1, -1], &[-1],             &[ 1, -1, -1, -1, -2]),
        (&[-1/2 + 1],       &[ 2],             &[ 0,  1]),
        (&[0, -1/2 + 1],    &[ 2],             &[ 0,  0,  1]),
        (&[ 1,  2],         &[ 1,  2,  3],     &[1, 4,  7,  6]),
        (&[-1, -1],         &[-1, -1, -1],     &[1, 0, -1, -2, -1]),
        (&[-1, -1, -1],     &[-1, -1, -1, -1], &[1, 0,  0, -1, -2, -1, -1]),
        (&[ 0,  0,  1],     &[ 1,  2,  3],     &[0, 0,  1,  2,  3]),
        (&[ 0,  0,  1],     &[ 0,  0,  0,  1], &[0, 0,  0,  0,  0,  1])
    ];

    static div_rem_quadruples: &'static [(&'static [BigDigit],
                                           &'static [BigDigit],
                                           &'static [BigDigit],
                                           &'static [BigDigit])]
        = &[
            (&[ 1],        &[ 2], &[],               &[1]),
            (&[ 1,  1],    &[ 2], &[-1/2+1],         &[1]),
            (&[ 1,  1, 1], &[ 2], &[-1/2+1, -1/2+1], &[1]),
            (&[ 0,  1],    &[-1], &[1],              &[1]),
            (&[-1, -1],    &[-2], &[2, 1],           &[3])
        ];

    #[test]
    fn test_mul() {
        for elm in mul_triples.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);

            assert!(a * b == c);
            assert!(b * a == c);
        }

        for elm in div_rem_quadruples.iter() {
            let (a_vec, b_vec, c_vec, d_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);
            let d = BigUint::from_slice(d_vec);

            assert!(a == b * c + d);
            assert!(a == c * b + d);
        }
    }

    #[test]
    fn test_div_rem() {
        for elm in mul_triples.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);

            if !a.is_zero() {
                assert_eq!(c.div_rem(&a), (b.clone(), Zero::zero()));
            }
            if !b.is_zero() {
                assert_eq!(c.div_rem(&b), (a.clone(), Zero::zero()));
            }
        }

        for elm in div_rem_quadruples.iter() {
            let (a_vec, b_vec, c_vec, d_vec) = *elm;
            let a = BigUint::from_slice(a_vec);
            let b = BigUint::from_slice(b_vec);
            let c = BigUint::from_slice(c_vec);
            let d = BigUint::from_slice(d_vec);

            if !b.is_zero() { assert!(a.div_rem(&b) == (c, d)); }
        }
    }

    #[test]
    fn test_checked_add() {
        for elm in sum_triples.iter() {
            let (aVec, bVec, cVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);

            assert!(a.checked_add(&b).unwrap() == c);
            assert!(b.checked_add(&a).unwrap() == c);
        }
    }

    #[test]
    fn test_checked_sub() {
        for elm in sum_triples.iter() {
            let (aVec, bVec, cVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);

            assert!(c.checked_sub(&a).unwrap() == b);
            assert!(c.checked_sub(&b).unwrap() == a);

            if a > c {
                assert!(a.checked_sub(&c).is_none());
            }
            if b > c {
                assert!(b.checked_sub(&c).is_none());
            }
        }
    }

    #[test]
    fn test_checked_mul() {
        for elm in mul_triples.iter() {
            let (aVec, bVec, cVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);

            assert!(a.checked_mul(&b).unwrap() == c);
            assert!(b.checked_mul(&a).unwrap() == c);
        }

        for elm in div_rem_quadruples.iter() {
            let (aVec, bVec, cVec, dVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);
            let d = BigUint::from_slice(dVec);

            assert!(a == b.checked_mul(&c).unwrap() + d);
            assert!(a == c.checked_mul(&b).unwrap() + d);
        }
    }

    #[test]
    fn test_checked_div() {
        for elm in mul_triples.iter() {
            let (aVec, bVec, cVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);

            if !a.is_zero() {
                assert!(c.checked_div(&a).unwrap() == b);
            }
            if !b.is_zero() {
                assert!(c.checked_div(&b).unwrap() == a);
            }

            assert!(c.checked_div(&Zero::zero()).is_none());
        }
    }

    #[test]
    fn test_gcd() {
        fn check(a: uint, b: uint, c: uint) {
            let big_a: BigUint = FromPrimitive::from_uint(a).unwrap();
            let big_b: BigUint = FromPrimitive::from_uint(b).unwrap();
            let big_c: BigUint = FromPrimitive::from_uint(c).unwrap();

            assert_eq!(big_a.gcd(&big_b), big_c);
        }

        check(10, 2, 2);
        check(10, 3, 1);
        check(0, 3, 3);
        check(3, 3, 3);
        check(56, 42, 14);
    }

    #[test]
    fn test_lcm() {
        fn check(a: uint, b: uint, c: uint) {
            let big_a: BigUint = FromPrimitive::from_uint(a).unwrap();
            let big_b: BigUint = FromPrimitive::from_uint(b).unwrap();
            let big_c: BigUint = FromPrimitive::from_uint(c).unwrap();

            assert_eq!(big_a.lcm(&big_b), big_c);
        }

        check(1, 0, 0);
        check(0, 1, 0);
        check(1, 1, 1);
        check(8, 9, 72);
        check(11, 5, 55);
        check(99, 17, 1683);
    }

    #[test]
    fn test_is_even() {
        let one: BigUint = FromStr::from_str("1").unwrap();
        let two: BigUint = FromStr::from_str("2").unwrap();
        let thousand: BigUint = FromStr::from_str("1000").unwrap();
        let big: BigUint = FromStr::from_str("1000000000000000000000").unwrap();
        let bigger: BigUint = FromStr::from_str("1000000000000000000001").unwrap();
        assert!(one.is_odd());
        assert!(two.is_even());
        assert!(thousand.is_even());
        assert!(big.is_even());
        assert!(bigger.is_odd());
        assert!((one << 64).is_even());
        assert!(((one << 64) + one).is_odd());
    }

    fn to_str_pairs() -> Vec<(BigUint, Vec<(uint, String)>)> {
        let bits = BigDigit::bits;
        vec!(( Zero::zero(), vec!(
            (2, "0".to_string()), (3, "0".to_string())
        )), ( BigUint::from_slice([ 0xff ]), vec!(
            (2,  "11111111".to_string()),
            (3,  "100110".to_string()),
            (4,  "3333".to_string()),
            (5,  "2010".to_string()),
            (6,  "1103".to_string()),
            (7,  "513".to_string()),
            (8,  "377".to_string()),
            (9,  "313".to_string()),
            (10, "255".to_string()),
            (11, "212".to_string()),
            (12, "193".to_string()),
            (13, "168".to_string()),
            (14, "143".to_string()),
            (15, "120".to_string()),
            (16, "ff".to_string())
        )), ( BigUint::from_slice([ 0xfff ]), vec!(
            (2,  "111111111111".to_string()),
            (4,  "333333".to_string()),
            (16, "fff".to_string())
        )), ( BigUint::from_slice([ 1, 2 ]), vec!(
            (2,
             format!("10{}1", "0".repeat(bits - 1))),
            (4,
             format!("2{}1", "0".repeat(bits / 2 - 1))),
            (10, match bits {
                32 => "8589934593".to_string(),
                16 => "131073".to_string(),
                _ => fail!()
            }),
            (16,
             format!("2{}1", "0".repeat(bits / 4 - 1)))
        )), ( BigUint::from_slice([ 1, 2, 3 ]), vec!(
            (2,
             format!("11{}10{}1",
                     "0".repeat(bits - 2),
                     "0".repeat(bits - 1))),
            (4,
             format!("3{}2{}1",
                     "0".repeat(bits / 2 - 1),
                     "0".repeat(bits / 2 - 1))),
            (10, match bits {
                32 => "55340232229718589441".to_string(),
                16 => "12885032961".to_string(),
                _ => fail!()
            }),
            (16,
             format!("3{}2{}1",
                     "0".repeat(bits / 4 - 1),
                     "0".repeat(bits / 4 - 1)))
        )) )
    }

    #[test]
    fn test_to_str_radix() {
        let r = to_str_pairs();
        for num_pair in r.iter() {
            let &(ref n, ref rs) = num_pair;
            for str_pair in rs.iter() {
                let &(ref radix, ref str) = str_pair;
                assert_eq!(n.to_str_radix(*radix).as_slice(),
                           str.as_slice());
            }
        }
    }

    #[test]
    fn test_from_str_radix() {
        let r = to_str_pairs();
        for num_pair in r.iter() {
            let &(ref n, ref rs) = num_pair;
            for str_pair in rs.iter() {
                let &(ref radix, ref str) = str_pair;
                assert_eq!(n,
                           &FromStrRadix::from_str_radix(str.as_slice(),
                                                         *radix).unwrap());
            }
        }

        let zed: Option<BigUint> = FromStrRadix::from_str_radix("Z", 10);
        assert_eq!(zed, None);
        let blank: Option<BigUint> = FromStrRadix::from_str_radix("_", 2);
        assert_eq!(blank, None);
        let minus_one: Option<BigUint> = FromStrRadix::from_str_radix("-1",
                                                                      10);
        assert_eq!(minus_one, None);
    }

    #[test]
    fn test_factor() {
        fn factor(n: uint) -> BigUint {
            let mut f: BigUint = One::one();
            for i in range(2, n + 1) {
                // FIXME(#5992): assignment operator overloads
                // f *= FromPrimitive::from_uint(i);
                f = f * FromPrimitive::from_uint(i).unwrap();
            }
            return f;
        }

        fn check(n: uint, s: &str) {
            let n = factor(n);
            let ans = match FromStrRadix::from_str_radix(s, 10) {
                Some(x) => x, None => fail!()
            };
            assert_eq!(n, ans);
        }

        check(3, "6");
        check(10, "3628800");
        check(20, "2432902008176640000");
        check(30, "265252859812191058636308480000000");
    }

    #[test]
    fn test_bits() {
        assert_eq!(BigUint::new(vec!(0,0,0,0)).bits(), 0);
        let n: BigUint = FromPrimitive::from_uint(0).unwrap();
        assert_eq!(n.bits(), 0);
        let n: BigUint = FromPrimitive::from_uint(1).unwrap();
        assert_eq!(n.bits(), 1);
        let n: BigUint = FromPrimitive::from_uint(3).unwrap();
        assert_eq!(n.bits(), 2);
        let n: BigUint = FromStrRadix::from_str_radix("4000000000", 16).unwrap();
        assert_eq!(n.bits(), 39);
        let one: BigUint = One::one();
        assert_eq!((one << 426).bits(), 427);
    }

    #[test]
    fn test_rand() {
        let mut rng = task_rng();
        let _n: BigUint = rng.gen_biguint(137);
        assert!(rng.gen_biguint(0).is_zero());
    }

    #[test]
    fn test_rand_range() {
        let mut rng = task_rng();

        for _ in range(0u, 10) {
            assert_eq!(rng.gen_bigint_range(&FromPrimitive::from_uint(236).unwrap(),
                                            &FromPrimitive::from_uint(237).unwrap()),
                       FromPrimitive::from_uint(236).unwrap());
        }

        let l = FromPrimitive::from_uint(403469000 + 2352).unwrap();
        let u = FromPrimitive::from_uint(403469000 + 3513).unwrap();
        for _ in range(0u, 1000) {
            let n: BigUint = rng.gen_biguint_below(&u);
            assert!(n < u);

            let n: BigUint = rng.gen_biguint_range(&l, &u);
            assert!(n >= l);
            assert!(n < u);
        }
    }

    #[test]
    #[should_fail]
    fn test_zero_rand_range() {
        task_rng().gen_biguint_range(&FromPrimitive::from_uint(54).unwrap(),
                                     &FromPrimitive::from_uint(54).unwrap());
    }

    #[test]
    #[should_fail]
    fn test_negative_rand_range() {
        let mut rng = task_rng();
        let l = FromPrimitive::from_uint(2352).unwrap();
        let u = FromPrimitive::from_uint(3513).unwrap();
        // Switching u and l should fail:
        let _n: BigUint = rng.gen_biguint_range(&u, &l);
    }
}

#[cfg(test)]
mod bigint_tests {
    use Integer;
    use super::{BigDigit, BigUint, ToBigUint};
    use super::{Sign, Minus, Zero, Plus, BigInt, RandBigInt, ToBigInt};

    use std::cmp::{Less, Equal, Greater};
    use std::i64;
    use std::num::CheckedDiv;
    use std::num::{Zero, One, FromStrRadix, ToStrRadix};
    use std::num::{ToPrimitive, FromPrimitive};
    use std::rand::task_rng;
    use std::u64;

    #[test]
    fn test_from_biguint() {
        fn check(inp_s: Sign, inp_n: uint, ans_s: Sign, ans_n: uint) {
            let inp = BigInt::from_biguint(inp_s, FromPrimitive::from_uint(inp_n).unwrap());
            let ans = BigInt { sign: ans_s, data: FromPrimitive::from_uint(ans_n).unwrap()};
            assert_eq!(inp, ans);
        }
        check(Plus, 1, Plus, 1);
        check(Plus, 0, Zero, 0);
        check(Minus, 1, Minus, 1);
        check(Zero, 1, Zero, 0);
    }

    #[test]
    fn test_cmp() {
        let vs = [ &[2 as BigDigit], &[1, 1], &[2, 1], &[1, 1, 1] ];
        let mut nums = Vec::new();
        for s in vs.iter().rev() {
            nums.push(BigInt::from_slice(Minus, *s));
        }
        nums.push(Zero::zero());
        nums.extend(vs.iter().map(|s| BigInt::from_slice(Plus, *s)));

        for (i, ni) in nums.iter().enumerate() {
            for (j0, nj) in nums.slice(i, nums.len()).iter().enumerate() {
                let j = i + j0;
                if i == j {
                    assert_eq!(ni.cmp(nj), Equal);
                    assert_eq!(nj.cmp(ni), Equal);
                    assert_eq!(ni, nj);
                    assert!(!(ni != nj));
                    assert!(ni <= nj);
                    assert!(ni >= nj);
                    assert!(!(ni < nj));
                    assert!(!(ni > nj));
                } else {
                    assert_eq!(ni.cmp(nj), Less);
                    assert_eq!(nj.cmp(ni), Greater);

                    assert!(!(ni == nj));
                    assert!(ni != nj);

                    assert!(ni <= nj);
                    assert!(!(ni >= nj));
                    assert!(ni < nj);
                    assert!(!(ni > nj));

                    assert!(!(nj <= ni));
                    assert!(nj >= ni);
                    assert!(!(nj < ni));
                    assert!(nj > ni);
                }
            }
        }
    }

    #[test]
    fn test_convert_i64() {
        fn check(b1: BigInt, i: i64) {
            let b2: BigInt = FromPrimitive::from_i64(i).unwrap();
            assert!(b1 == b2);
            assert!(b1.to_i64().unwrap() == i);
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);
        check(i64::MIN.to_bigint().unwrap(), i64::MIN);
        check(i64::MAX.to_bigint().unwrap(), i64::MAX);

        assert_eq!(
            (i64::MAX as u64 + 1).to_bigint().unwrap().to_i64(),
            None);

        assert_eq!(
            BigInt::from_biguint(Plus,  BigUint::new(vec!(1, 2, 3, 4, 5))).to_i64(),
            None);

        assert_eq!(
            BigInt::from_biguint(Minus, BigUint::new(vec!(1,0,0,1<<(BigDigit::bits-1)))).to_i64(),
            None);

        assert_eq!(
            BigInt::from_biguint(Minus, BigUint::new(vec!(1, 2, 3, 4, 5))).to_i64(),
            None);
    }

    #[test]
    fn test_convert_u64() {
        fn check(b1: BigInt, u: u64) {
            let b2: BigInt = FromPrimitive::from_u64(u).unwrap();
            assert!(b1 == b2);
            assert!(b1.to_u64().unwrap() == u);
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);
        check(u64::MIN.to_bigint().unwrap(), u64::MIN);
        check(u64::MAX.to_bigint().unwrap(), u64::MAX);

        assert_eq!(
            BigInt::from_biguint(Plus, BigUint::new(vec!(1, 2, 3, 4, 5))).to_u64(),
            None);

        let max_value: BigUint = FromPrimitive::from_u64(u64::MAX).unwrap();
        assert_eq!(BigInt::from_biguint(Minus, max_value).to_u64(), None);
        assert_eq!(BigInt::from_biguint(Minus, BigUint::new(vec!(1, 2, 3, 4, 5))).to_u64(), None);
    }

    #[test]
    fn test_convert_to_biguint() {
        fn check(n: BigInt, ans_1: BigUint) {
            assert_eq!(n.to_biguint().unwrap(), ans_1);
            assert_eq!(n.to_biguint().unwrap().to_bigint().unwrap(), n);
        }
        let zero: BigInt = Zero::zero();
        let unsigned_zero: BigUint = Zero::zero();
        let positive = BigInt::from_biguint(
            Plus, BigUint::new(vec!(1,2,3)));
        let negative = -positive;

        check(zero, unsigned_zero);
        check(positive, BigUint::new(vec!(1,2,3)));

        assert_eq!(negative.to_biguint(), None);
    }

    static sum_triples: &'static [(&'static [BigDigit],
                                   &'static [BigDigit],
                                   &'static [BigDigit])] = &[
        (&[],          &[],       &[]),
        (&[],          &[ 1],     &[ 1]),
        (&[ 1],        &[ 1],     &[ 2]),
        (&[ 1],        &[ 1,  1], &[ 2,  1]),
        (&[ 1],        &[-1],     &[ 0,  1]),
        (&[ 1],        &[-1, -1], &[ 0,  0, 1]),
        (&[-1, -1],    &[-1, -1], &[-2, -1, 1]),
        (&[ 1,  1, 1], &[-1, -1], &[ 0,  1, 2]),
        (&[ 2,  2, 1], &[-1, -2], &[ 1,  1, 2])
    ];

    #[test]
    fn test_add() {
        for elm in sum_triples.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);

            assert!(a + b == c);
            assert!(b + a == c);
            assert!(c + (-a) == b);
            assert!(c + (-b) == a);
            assert!(a + (-c) == (-b));
            assert!(b + (-c) == (-a));
            assert!((-a) + (-b) == (-c))
            assert!(a + (-a) == Zero::zero());
        }
    }

    #[test]
    fn test_sub() {
        for elm in sum_triples.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);

            assert!(c - a == b);
            assert!(c - b == a);
            assert!((-b) - a == (-c))
            assert!((-a) - b == (-c))
            assert!(b - (-a) == c);
            assert!(a - (-b) == c);
            assert!((-c) - (-a) == (-b));
            assert!(a - a == Zero::zero());
        }
    }

    static mul_triples: &'static [(&'static [BigDigit],
                                   &'static [BigDigit],
                                   &'static [BigDigit])] = &[
        (&[],               &[],               &[]),
        (&[],               &[ 1],             &[]),
        (&[ 2],             &[],               &[]),
        (&[ 1],             &[ 1],             &[1]),
        (&[ 2],             &[ 3],             &[ 6]),
        (&[ 1],             &[ 1,  1,  1],     &[1, 1,  1]),
        (&[ 1,  2,  3],     &[ 3],             &[ 3,  6,  9]),
        (&[ 1,  1,  1],     &[-1],             &[-1, -1, -1]),
        (&[ 1,  2,  3],     &[-1],             &[-1, -2, -2, 2]),
        (&[ 1,  2,  3,  4], &[-1],             &[-1, -2, -2, -2, 3]),
        (&[-1],             &[-1],             &[ 1, -2]),
        (&[-1, -1],         &[-1],             &[ 1, -1, -2]),
        (&[-1, -1, -1],     &[-1],             &[ 1, -1, -1, -2]),
        (&[-1, -1, -1, -1], &[-1],             &[ 1, -1, -1, -1, -2]),
        (&[-1/2 + 1],       &[ 2],             &[ 0,  1]),
        (&[0, -1/2 + 1],    &[ 2],             &[ 0,  0,  1]),
        (&[ 1,  2],         &[ 1,  2,  3],     &[1, 4,  7,  6]),
        (&[-1, -1],         &[-1, -1, -1],     &[1, 0, -1, -2, -1]),
        (&[-1, -1, -1],     &[-1, -1, -1, -1], &[1, 0,  0, -1, -2, -1, -1]),
        (&[ 0,  0,  1],     &[ 1,  2,  3],     &[0, 0,  1,  2,  3]),
        (&[ 0,  0,  1],     &[ 0,  0,  0,  1], &[0, 0,  0,  0,  0,  1])
    ];

    static div_rem_quadruples: &'static [(&'static [BigDigit],
                                          &'static [BigDigit],
                                          &'static [BigDigit],
                                          &'static [BigDigit])]
        = &[
            (&[ 1],        &[ 2], &[],               &[1]),
            (&[ 1,  1],    &[ 2], &[-1/2+1],         &[1]),
            (&[ 1,  1, 1], &[ 2], &[-1/2+1, -1/2+1], &[1]),
            (&[ 0,  1],    &[-1], &[1],              &[1]),
            (&[-1, -1],    &[-2], &[2, 1],           &[3])
        ];

    #[test]
    fn test_mul() {
        for elm in mul_triples.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);

            assert!(a * b == c);
            assert!(b * a == c);

            assert!((-a) * b == -c);
            assert!((-b) * a == -c);
        }

        for elm in div_rem_quadruples.iter() {
            let (a_vec, b_vec, c_vec, d_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);
            let d = BigInt::from_slice(Plus, d_vec);

            assert!(a == b * c + d);
            assert!(a == c * b + d);
        }
    }

    #[test]
    fn test_div_mod_floor() {
        fn check_sub(a: &BigInt, b: &BigInt, ans_d: &BigInt, ans_m: &BigInt) {
            let (d, m) = a.div_mod_floor(b);
            if !m.is_zero() {
                assert_eq!(m.sign, b.sign);
            }
            assert!(m.abs() <= b.abs());
            assert!(*a == b * d + m);
            assert!(d == *ans_d);
            assert!(m == *ans_m);
        }

        fn check(a: &BigInt, b: &BigInt, d: &BigInt, m: &BigInt) {
            if m.is_zero() {
                check_sub(a, b, d, m);
                check_sub(a, &b.neg(), &d.neg(), m);
                check_sub(&a.neg(), b, &d.neg(), m);
                check_sub(&a.neg(), &b.neg(), d, m);
            } else {
                check_sub(a, b, d, m);
                check_sub(a, &b.neg(), &(d.neg() - One::one()), &(m - *b));
                check_sub(&a.neg(), b, &(d.neg() - One::one()), &(b - *m));
                check_sub(&a.neg(), &b.neg(), d, &m.neg());
            }
        }

        for elm in mul_triples.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);

            if !a.is_zero() { check(&c, &a, &b, &Zero::zero()); }
            if !b.is_zero() { check(&c, &b, &a, &Zero::zero()); }
        }

        for elm in div_rem_quadruples.iter() {
            let (a_vec, b_vec, c_vec, d_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);
            let d = BigInt::from_slice(Plus, d_vec);

            if !b.is_zero() {
                check(&a, &b, &c, &d);
            }
        }
    }


    #[test]
    fn test_div_rem() {
        fn check_sub(a: &BigInt, b: &BigInt, ans_q: &BigInt, ans_r: &BigInt) {
            let (q, r) = a.div_rem(b);
            if !r.is_zero() {
                assert_eq!(r.sign, a.sign);
            }
            assert!(r.abs() <= b.abs());
            assert!(*a == b * q + r);
            assert!(q == *ans_q);
            assert!(r == *ans_r);
        }

        fn check(a: &BigInt, b: &BigInt, q: &BigInt, r: &BigInt) {
            check_sub(a, b, q, r);
            check_sub(a, &b.neg(), &q.neg(), r);
            check_sub(&a.neg(), b, &q.neg(), &r.neg());
            check_sub(&a.neg(), &b.neg(), q, &r.neg());
        }
        for elm in mul_triples.iter() {
            let (a_vec, b_vec, c_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);

            if !a.is_zero() { check(&c, &a, &b, &Zero::zero()); }
            if !b.is_zero() { check(&c, &b, &a, &Zero::zero()); }
        }

        for elm in div_rem_quadruples.iter() {
            let (a_vec, b_vec, c_vec, d_vec) = *elm;
            let a = BigInt::from_slice(Plus, a_vec);
            let b = BigInt::from_slice(Plus, b_vec);
            let c = BigInt::from_slice(Plus, c_vec);
            let d = BigInt::from_slice(Plus, d_vec);

            if !b.is_zero() {
                check(&a, &b, &c, &d);
            }
        }
    }

    #[test]
    fn test_checked_add() {
        for elm in sum_triples.iter() {
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

            assert!(a.checked_add(&b).unwrap() == c);
            assert!(b.checked_add(&a).unwrap() == c);
            assert!(c.checked_add(&(-a)).unwrap() == b);
            assert!(c.checked_add(&(-b)).unwrap() == a);
            assert!(a.checked_add(&(-c)).unwrap() == (-b));
            assert!(b.checked_add(&(-c)).unwrap() == (-a));
            assert!((-a).checked_add(&(-b)).unwrap() == (-c))
            assert!(a.checked_add(&(-a)).unwrap() == Zero::zero());
        }
    }

    #[test]
    fn test_checked_sub() {
        for elm in sum_triples.iter() {
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

            assert!(c.checked_sub(&a).unwrap() == b);
            assert!(c.checked_sub(&b).unwrap() == a);
            assert!((-b).checked_sub(&a).unwrap() == (-c))
            assert!((-a).checked_sub(&b).unwrap() == (-c))
            assert!(b.checked_sub(&(-a)).unwrap() == c);
            assert!(a.checked_sub(&(-b)).unwrap() == c);
            assert!((-c).checked_sub(&(-a)).unwrap() == (-b));
            assert!(a.checked_sub(&a).unwrap() == Zero::zero());
        }
    }

    #[test]
    fn test_checked_mul() {
        for elm in mul_triples.iter() {
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

            assert!(a.checked_mul(&b).unwrap() == c);
            assert!(b.checked_mul(&a).unwrap() == c);

            assert!((-a).checked_mul(&b).unwrap() == -c);
            assert!((-b).checked_mul(&a).unwrap() == -c);
        }

        for elm in div_rem_quadruples.iter() {
            let (aVec, bVec, cVec, dVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);
            let d = BigInt::from_slice(Plus, dVec);

            assert!(a == b.checked_mul(&c).unwrap() + d);
            assert!(a == c.checked_mul(&b).unwrap() + d);
        }
    }
    #[test]
    fn test_checked_div() {
        for elm in mul_triples.iter() {
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

            if !a.is_zero() {
                assert!(c.checked_div(&a).unwrap() == b);
                assert!((-c).checked_div(&(-a)).unwrap() == b);
                assert!((-c).checked_div(&a).unwrap() == -b);
            }
            if !b.is_zero() {
                assert!(c.checked_div(&b).unwrap() == a);
                assert!((-c).checked_div(&(-b)).unwrap() == a);
                assert!((-c).checked_div(&b).unwrap() == -a);
            }

            assert!(c.checked_div(&Zero::zero()).is_none());
            assert!((-c).checked_div(&Zero::zero()).is_none());
        }
    }

    #[test]
    fn test_gcd() {
        fn check(a: int, b: int, c: int) {
            let big_a: BigInt = FromPrimitive::from_int(a).unwrap();
            let big_b: BigInt = FromPrimitive::from_int(b).unwrap();
            let big_c: BigInt = FromPrimitive::from_int(c).unwrap();

            assert_eq!(big_a.gcd(&big_b), big_c);
        }

        check(10, 2, 2);
        check(10, 3, 1);
        check(0, 3, 3);
        check(3, 3, 3);
        check(56, 42, 14);
        check(3, -3, 3);
        check(-6, 3, 3);
        check(-4, -2, 2);
    }

    #[test]
    fn test_lcm() {
        fn check(a: int, b: int, c: int) {
            let big_a: BigInt = FromPrimitive::from_int(a).unwrap();
            let big_b: BigInt = FromPrimitive::from_int(b).unwrap();
            let big_c: BigInt = FromPrimitive::from_int(c).unwrap();

            assert_eq!(big_a.lcm(&big_b), big_c);
        }

        check(1, 0, 0);
        check(0, 1, 0);
        check(1, 1, 1);
        check(-1, 1, 1);
        check(1, -1, 1);
        check(-1, -1, 1);
        check(8, 9, 72);
        check(11, 5, 55);
    }

    #[test]
    fn test_abs_sub() {
        let zero: BigInt = Zero::zero();
        let one: BigInt = One::one();
        assert_eq!((-one).abs_sub(&one), zero);
        let one: BigInt = One::one();
        let zero: BigInt = Zero::zero();
        assert_eq!(one.abs_sub(&one), zero);
        let one: BigInt = One::one();
        let zero: BigInt = Zero::zero();
        assert_eq!(one.abs_sub(&zero), one);
        let one: BigInt = One::one();
        let two: BigInt = FromPrimitive::from_int(2).unwrap();
        assert_eq!(one.abs_sub(&-one), two);
    }

    #[test]
    fn test_to_str_radix() {
        fn check(n: int, ans: &str) {
            let n: BigInt = FromPrimitive::from_int(n).unwrap();
            assert!(ans == n.to_str_radix(10).as_slice());
        }
        check(10, "10");
        check(1, "1");
        check(0, "0");
        check(-1, "-1");
        check(-10, "-10");
    }


    #[test]
    fn test_from_str_radix() {
        fn check(s: &str, ans: Option<int>) {
            let ans = ans.map(|n| {
                let x: BigInt = FromPrimitive::from_int(n).unwrap();
                x
            });
            assert_eq!(FromStrRadix::from_str_radix(s, 10), ans);
        }
        check("10", Some(10));
        check("1", Some(1));
        check("0", Some(0));
        check("-1", Some(-1));
        check("-10", Some(-10));
        check("Z", None);
        check("_", None);

        // issue 10522, this hit an edge case that caused it to
        // attempt to allocate a vector of size (-1u) == huge.
        let x: BigInt =
            from_str(format!("1{}", "0".repeat(36)).as_slice()).unwrap();
        let _y = x.to_string();
    }

    #[test]
    fn test_neg() {
        assert!(-BigInt::new(Plus,  vec!(1, 1, 1)) ==
            BigInt::new(Minus, vec!(1, 1, 1)));
        assert!(-BigInt::new(Minus, vec!(1, 1, 1)) ==
            BigInt::new(Plus,  vec!(1, 1, 1)));
        let zero: BigInt = Zero::zero();
        assert_eq!(-zero, zero);
    }

    #[test]
    fn test_rand() {
        let mut rng = task_rng();
        let _n: BigInt = rng.gen_bigint(137);
        assert!(rng.gen_bigint(0).is_zero());
    }

    #[test]
    fn test_rand_range() {
        let mut rng = task_rng();

        for _ in range(0u, 10) {
            assert_eq!(rng.gen_bigint_range(&FromPrimitive::from_uint(236).unwrap(),
                                            &FromPrimitive::from_uint(237).unwrap()),
                       FromPrimitive::from_uint(236).unwrap());
        }

        fn check(l: BigInt, u: BigInt) {
            let mut rng = task_rng();
            for _ in range(0u, 1000) {
                let n: BigInt = rng.gen_bigint_range(&l, &u);
                assert!(n >= l);
                assert!(n < u);
            }
        }
        let l: BigInt = FromPrimitive::from_uint(403469000 + 2352).unwrap();
        let u: BigInt = FromPrimitive::from_uint(403469000 + 3513).unwrap();
        check( l.clone(),  u.clone());
        check(-l.clone(),  u.clone());
        check(-u.clone(), -l.clone());
    }

    #[test]
    #[should_fail]
    fn test_zero_rand_range() {
        task_rng().gen_bigint_range(&FromPrimitive::from_int(54).unwrap(),
                                    &FromPrimitive::from_int(54).unwrap());
    }

    #[test]
    #[should_fail]
    fn test_negative_rand_range() {
        let mut rng = task_rng();
        let l = FromPrimitive::from_uint(2352).unwrap();
        let u = FromPrimitive::from_uint(3513).unwrap();
        // Switching u and l should fail:
        let _n: BigInt = rng.gen_bigint_range(&u, &l);
    }
}

#[cfg(test)]
mod bench {
    extern crate test;
    use self::test::Bencher;
    use super::BigUint;
    use std::iter;
    use std::mem::replace;
    use std::num::{FromPrimitive, Zero, One};

    fn factorial(n: uint) -> BigUint {
        let mut f: BigUint = One::one();
        for i in iter::range_inclusive(1, n) {
            f = f * FromPrimitive::from_uint(i).unwrap();
        }
        f
    }

    fn fib(n: uint) -> BigUint {
        let mut f0: BigUint = Zero::zero();
        let mut f1: BigUint = One::one();
        for _ in range(0, n) {
            let f2 = f0 + f1;
            f0 = replace(&mut f1, f2);
        }
        f0
    }

    #[bench]
    fn factorial_100(b: &mut Bencher) {
        b.iter(|| {
            factorial(100);
        });
    }

    #[bench]
    fn fib_100(b: &mut Bencher) {
        b.iter(|| {
            fib(100);
        });
    }

    #[bench]
    fn to_string(b: &mut Bencher) {
        let fac = factorial(100);
        let fib = fib(100);
        b.iter(|| {
            fac.to_string();
        });
        b.iter(|| {
            fib.to_string();
        });
    }

    #[bench]
    fn shr(b: &mut Bencher) {
        let n = { let one : BigUint = One::one(); one << 1000 };
        b.iter(|| {
            let mut m = n.clone();
            for _ in range(0u, 10) {
                m = m >> 1;
            }
        })
    }
}
