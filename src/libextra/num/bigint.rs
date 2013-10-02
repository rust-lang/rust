// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

A Big integer (signed version: `BigInt`, unsigned version: `BigUint`).

A `BigUint` is represented as an array of `BigDigit`s.
A `BigInt` is a combination of `BigUint` and `Sign`.
*/

#[allow(missing_doc)];
#[allow(non_uppercase_statics)];

use std::cmp::{Eq, Ord, TotalEq, TotalOrd, Ordering, Less, Equal, Greater};
use std::num;
use std::num::{Zero, One, ToStrRadix, FromStrRadix, Orderable};
use std::num::{ToPrimitive, FromPrimitive};
use std::rand::Rng;
use std::str;
use std::uint;
use std::{i64, u64};
use std::vec;

/**
A `BigDigit` is a `BigUint`'s composing element.

A `BigDigit` is half the size of machine word size.
*/
#[cfg(target_word_size = "32")]
pub type BigDigit = u16;

/**
A `BigDigit` is a `BigUint`'s composing element.

A `BigDigit` is half the size of machine word size.
*/
#[cfg(target_word_size = "64")]
pub type BigDigit = u32;

pub static ZERO_BIG_DIGIT: BigDigit = 0;

pub mod BigDigit {
    use bigint::BigDigit;

    #[cfg(target_word_size = "32")]
    pub static bits: uint = 16;

    #[cfg(target_word_size = "64")]
    pub static bits: uint = 32;

    pub static base: uint = 1 << bits;
    static hi_mask: uint = (-1 as uint) << bits;
    static lo_mask: uint = (-1 as uint) >> bits;

    #[inline]
    fn get_hi(n: uint) -> BigDigit { (n >> bits) as BigDigit }
    #[inline]
    fn get_lo(n: uint) -> BigDigit { (n & lo_mask) as BigDigit }

    /// Split one machine sized unsigned integer into two `BigDigit`s.
    #[inline]
    pub fn from_uint(n: uint) -> (BigDigit, BigDigit) {
        (get_hi(n), get_lo(n))
    }

    /// Join two `BigDigit`s into one machine sized unsigned integer
    #[inline]
    pub fn to_uint(hi: BigDigit, lo: BigDigit) -> uint {
        (lo as uint) | ((hi as uint) << bits)
    }
}

/**
A big unsigned integer type.

A `BigUint`-typed value `BigUint { data: @[a, b, c] }` represents a number
`(a + b * BigDigit::base + c * BigDigit::base^2)`.
*/
#[deriving(Clone)]
pub struct BigUint {
    priv data: ~[BigDigit]
}

impl Eq for BigUint {
    #[inline]
    fn eq(&self, other: &BigUint) -> bool { self.equals(other) }
}

impl TotalEq for BigUint {
    #[inline]
    fn equals(&self, other: &BigUint) -> bool {
        match self.cmp(other) { Equal => true, _ => false }
    }
}

impl Ord for BigUint {
    #[inline]
    fn lt(&self, other: &BigUint) -> bool {
        match self.cmp(other) { Less => true, _ => false}
    }
}

impl TotalOrd for BigUint {
    #[inline]
    fn cmp(&self, other: &BigUint) -> Ordering {
        let (s_len, o_len) = (self.data.len(), other.data.len());
        if s_len < o_len { return Less; }
        if s_len > o_len { return Greater;  }

        for (&self_i, &other_i) in self.data.rev_iter().zip(other.data.rev_iter()) {
            if self_i < other_i { return Less; }
            if self_i > other_i { return Greater; }
        }
        return Equal;
    }
}

impl ToStr for BigUint {
    #[inline]
    fn to_str(&self) -> ~str { self.to_str_radix(10) }
}

impl FromStr for BigUint {
    #[inline]
    fn from_str(s: &str) -> Option<BigUint> {
        FromStrRadix::from_str_radix(s, 10)
    }
}

impl Num for BigUint {}

impl Orderable for BigUint {
    #[inline]
    fn min(&self, other: &BigUint) -> BigUint {
        if self < other { self.clone() } else { other.clone() }
    }

    #[inline]
    fn max(&self, other: &BigUint) -> BigUint {
        if self > other { self.clone() } else { other.clone() }
    }

    #[inline]
    fn clamp(&self, mn: &BigUint, mx: &BigUint) -> BigUint {
        if self > mx { mx.clone() } else
        if self < mn { mn.clone() } else { self.clone() }
    }
}

impl BitAnd<BigUint, BigUint> for BigUint {
    fn bitand(&self, other: &BigUint) -> BigUint {
        let new_len = num::min(self.data.len(), other.data.len());
        let anded = do vec::from_fn(new_len) |i| {
            // i will never be less than the size of either data vector
            let ai = self.data[i];
            let bi = other.data[i];
            ai & bi
        };
        return BigUint::new(anded);
    }
}

impl BitOr<BigUint, BigUint> for BigUint {
    fn bitor(&self, other: &BigUint) -> BigUint {
        let new_len = num::max(self.data.len(), other.data.len());
        let ored = do vec::from_fn(new_len) |i| {
            let ai = if i < self.data.len()  { self.data[i]  } else { 0 };
            let bi = if i < other.data.len() { other.data[i] } else { 0 };
            ai | bi
        };
        return BigUint::new(ored);
    }
}

impl BitXor<BigUint, BigUint> for BigUint {
    fn bitxor(&self, other: &BigUint) -> BigUint {
        let new_len = num::max(self.data.len(), other.data.len());
        let xored = do vec::from_fn(new_len) |i| {
            let ai = if i < self.data.len()  { self.data[i]  } else { 0 };
            let bi = if i < other.data.len() { other.data[i] } else { 0 };
            ai ^ bi
        };
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
    fn zero() -> BigUint { BigUint::new(~[]) }

    #[inline]
    fn is_zero(&self) -> bool { self.data.is_empty() }
}

impl One for BigUint {
    #[inline]
    fn one() -> BigUint { BigUint::new(~[1]) }
}

impl Unsigned for BigUint {}

impl Add<BigUint, BigUint> for BigUint {
    fn add(&self, other: &BigUint) -> BigUint {
        let new_len = num::max(self.data.len(), other.data.len());

        let mut carry = 0;
        let mut sum = do vec::from_fn(new_len) |i| {
            let ai = if i < self.data.len()  { self.data[i]  } else { 0 };
            let bi = if i < other.data.len() { other.data[i] } else { 0 };
            let (hi, lo) = BigDigit::from_uint(
                (ai as uint) + (bi as uint) + (carry as uint)
            );
            carry = hi;
            lo
        };
        if carry != 0 { sum.push(carry); }
        return BigUint::new(sum);
    }
}

impl Sub<BigUint, BigUint> for BigUint {
    fn sub(&self, other: &BigUint) -> BigUint {
        let new_len = num::max(self.data.len(), other.data.len());

        let mut borrow = 0;
        let diff = do vec::from_fn(new_len) |i| {
            let ai = if i < self.data.len()  { self.data[i]  } else { 0 };
            let bi = if i < other.data.len() { other.data[i] } else { 0 };
            let (hi, lo) = BigDigit::from_uint(
                (BigDigit::base) +
                (ai as uint) - (bi as uint) - (borrow as uint)
            );
            /*
            hi * (base) + lo == 1*(base) + ai - bi - borrow
            => ai - bi - borrow < 0 <=> hi == 0
            */
            borrow = if hi == 0 { 1 } else { 0 };
            lo
        };

        assert_eq!(borrow, 0);     // <=> assert!((self >= other));
        return BigUint::new(diff);
    }
}

impl Mul<BigUint, BigUint> for BigUint {
    fn mul(&self, other: &BigUint) -> BigUint {
        if self.is_zero() || other.is_zero() { return Zero::zero(); }

        let (s_len, o_len) = (self.data.len(), other.data.len());
        if s_len == 1 { return mul_digit(other, self.data[0]);  }
        if o_len == 1 { return mul_digit(self,  other.data[0]); }

        // Using Karatsuba multiplication
        // (a1 * base + a0) * (b1 * base + b0)
        // = a1*b1 * base^2 +
        //   (a1*b1 + a0*b0 - (a1-b0)*(b1-a0)) * base +
        //   a0*b0
        let half_len = num::max(s_len, o_len) / 2;
        let (sHi, sLo) = cut_at(self,  half_len);
        let (oHi, oLo) = cut_at(other, half_len);

        let ll = sLo * oLo;
        let hh = sHi * oHi;
        let mm = {
            let (s1, n1) = sub_sign(sHi, sLo);
            let (s2, n2) = sub_sign(oHi, oLo);
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
            let mut prod = do a.data.iter().map |ai| {
                let (hi, lo) = BigDigit::from_uint(
                    (*ai as uint) * (n as uint) + (carry as uint)
                );
                carry = hi;
                lo
            }.collect::<~[BigDigit]>();
            if carry != 0 { prod.push(carry); }
            return BigUint::new(prod);
        }

        #[inline]
        fn cut_at(a: &BigUint, n: uint) -> (BigUint, BigUint) {
            let mid = num::min(a.data.len(), n);
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
    fn neg(&self) -> BigUint { fail2!() }
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
        if other.is_zero() { fail2!() }
        if self.is_zero() { return (Zero::zero(), Zero::zero()); }
        if *other == One::one() { return ((*self).clone(), Zero::zero()); }

        match self.cmp(other) {
            Less    => return (Zero::zero(), (*self).clone()),
            Equal   => return (One::one(), Zero::zero()),
            Greater => {} // Do nothing
        }

        let mut shift = 0;
        let mut n = *other.data.last();
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
                    // FIXME(#6050): overloaded operators force moves with generic types
                    // d0 -= d_unit
                    d0   = d0 - d_unit;
                    // FIXME(#6050): overloaded operators force moves with generic types
                    // prod = prod - b_unit;
                    prod = prod - b_unit
                }
                if d0.is_zero() {
                    n = 2;
                    continue;
                }
                n = 1;
                // FIXME(#6102): Assignment operator for BigInt causes ICE
                // d += d0;
                d = d + d0;
                // FIXME(#6102): Assignment operator for BigInt causes ICE
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

            let an = a.data.slice(a.data.len() - n, a.data.len());
            let bn = *b.data.last();
            let mut d = ~[];
            let mut carry = 0;
            for elt in an.rev_iter() {
                let ai = BigDigit::to_uint(carry, *elt);
                let di = ai / (bn as uint);
                assert!(di < BigDigit::base);
                carry = (ai % (bn as uint)) as BigDigit;
                d = ~[di as BigDigit] + d;
            }

            let shift = (a.data.len() - an.len()) - (b.data.len() - 1);
            if shift == 0 {
                return (BigUint::new(d), One::one(), (*b).clone());
            }
            let one: BigUint = One::one();
            return (BigUint::from_slice(d).shl_unit(shift),
                    one.shl_unit(shift),
                    b.shl_unit(shift));
        }
    }

    /**
     * Calculates the Greatest Common Divisor (GCD) of the number and `other`
     *
     * The result is always positive
     */
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

    /**
     * Calculates the Lowest Common Multiple (LCM) of the number and `other`
     */
    #[inline]
    fn lcm(&self, other: &BigUint) -> BigUint { ((*self * *other) / self.gcd(other)) }

    /// Returns `true` if the number can be divided by `other` without leaving a remainder
    #[inline]
    fn is_multiple_of(&self, other: &BigUint) -> bool { (*self % *other).is_zero() }

    /// Returns `true` if the number is divisible by `2`
    #[inline]
    fn is_even(&self) -> bool {
        // Considering only the last digit.
        if self.data.is_empty() {
            true
        } else {
            self.data[0].is_even()
        }
    }

    /// Returns `true` if the number is not divisible by `2`
    #[inline]
    fn is_odd(&self) -> bool { !self.is_even() }
}

impl ToPrimitive for BigUint {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        do self.to_u64().and_then |n| {
            // If top bit of u64 is set, it's too large to convert to i64.
            if n >> 63 == 0 {
                Some(n as i64)
            } else {
                None
            }
        }
    }

    #[cfg(target_word_size = "32")]
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        match self.data {
            [] => {
                Some(0)
            }
            [n0] => {
                Some(n0 as u64)
            }
            [n0, n1] => {
                Some(BigDigit::to_uint(n1, n0) as u64)
            }
            [n0, n1, n2] => {
                let n_lo = BigDigit::to_uint(n1, n0) as u64;
                let n_hi = n2 as u64;
                Some((n_hi << 32) + n_lo)
            }
            [n0, n1, n2, n3] => {
                let n_lo = BigDigit::to_uint(n1, n0) as u64;
                let n_hi = BigDigit::to_uint(n3, n2) as u64;
                Some((n_hi << 32) + n_lo)
            }
            _ => None
        }
    }

    #[cfg(target_word_size = "64")]
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        match self.data {
            [] => {
                Some(0)
            }
            [n0] => {
                Some(n0 as u64)
            }
            [n0, n1] => {
                Some(BigDigit::to_uint(n1, n0) as u64)
            }
            _ => None
        }
    }
}

impl FromPrimitive for BigUint {
    #[inline]
    fn from_i64(n: i64) -> Option<BigUint> {
        if (n > 0) {
            FromPrimitive::from_u64(n as u64)
        } else if (n == 0) {
            Some(Zero::zero())
        } else {
            None
        }
    }

    #[cfg(target_word_size = "32")]
    #[inline]
    fn from_u64(n: u64) -> Option<BigUint> {
        let n_lo = (n & 0x0000_0000_FFFF_FFFF) as uint;
        let n_hi = (n >> 32) as uint;

        let n = match (BigDigit::from_uint(n_hi), BigDigit::from_uint(n_lo)) {
            ((0,  0),  (0,  0))  => Zero::zero(),
            ((0,  0),  (0,  n0)) => BigUint::new(~[n0]),
            ((0,  0),  (n1, n0)) => BigUint::new(~[n0, n1]),
            ((0,  n2), (n1, n0)) => BigUint::new(~[n0, n1, n2]),
            ((n3, n2), (n1, n0)) => BigUint::new(~[n0, n1, n2, n3]),
        };
        Some(n)
    }

    #[cfg(target_word_size = "64")]
    #[inline]
    fn from_u64(n: u64) -> Option<BigUint> {
        let n = match BigDigit::from_uint(n as uint) {
            (0,  0)  => Zero::zero(),
            (0,  n0) => BigUint::new(~[n0]),
            (n1, n0) => BigUint::new(~[n0, n1])
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
    fn to_str_radix(&self, radix: uint) -> ~str {
        assert!(1 < radix && radix <= 16);
        let (base, max_len) = get_radix_base(radix);
        if base == BigDigit::base {
            return fill_concat(self.data, radix, max_len)
        }
        return fill_concat(convert_base((*self).clone(), base), radix, max_len);

        fn convert_base(n: BigUint, base: uint) -> ~[BigDigit] {
            let divider    = FromPrimitive::from_uint(base).unwrap();
            let mut result = ~[];
            let mut m      = n;
            while m > divider {
                let (d, m0) = m.div_mod_floor(&divider);
                result.push(m0.to_uint().unwrap() as BigDigit);
                m = d;
            }
            if !m.is_zero() {
                result.push(m.to_uint().unwrap() as BigDigit);
            }
            return result;
        }

        fn fill_concat(v: &[BigDigit], radix: uint, l: uint) -> ~str {
            if v.is_empty() { return ~"0" }
            let mut s = str::with_capacity(v.len() * l);
            for n in v.rev_iter() {
                let ss = (*n as uint).to_str_radix(radix);
                s.push_str("0".repeat(l - ss.len()));
                s.push_str(ss);
            }
            s.trim_left_chars(&'0').to_owned()
        }
    }
}

impl FromStrRadix for BigUint {
    /// Creates and initializes a `BigUint`.
    #[inline]
    fn from_str_radix(s: &str, radix: uint)
        -> Option<BigUint> {
        BigUint::parse_bytes(s.as_bytes(), radix)
    }
}

impl BigUint {
    /// Creates and initializes a `BigUint`.
    #[inline]
    pub fn new(v: ~[BigDigit]) -> BigUint {
        // omit trailing zeros
        let new_len = v.iter().rposition(|n| *n != 0).map_move_default(0, |p| p + 1);

        if new_len == v.len() { return BigUint { data: v }; }
        let mut v = v;
        v.truncate(new_len);
        return BigUint { data: v };
    }

    /// Creates and initializes a `BigUint`.
    #[inline]
    pub fn from_slice(slice: &[BigDigit]) -> BigUint {
        return BigUint::new(slice.to_owned());
    }

    /// Creates and initializes a `BigUint`.
    pub fn parse_bytes(buf: &[u8], radix: uint) -> Option<BigUint> {
        let (base, unit_len) = get_radix_base(radix);
        let base_num = match FromPrimitive::from_uint(base) {
            Some(base_num) => base_num,
            None => { return None; }
        };

        let mut end             = buf.len();
        let mut n: BigUint      = Zero::zero();
        let mut power: BigUint  = One::one();
        loop {
            let start = num::max(end, unit_len) - unit_len;
            match uint::parse_bytes(buf.slice(start, end), radix) {
                Some(d) => {
                    let d: Option<BigUint> = FromPrimitive::from_uint(d);
                    match d {
                        Some(d) => {
                            // FIXME(#6102): Assignment operator for BigInt
                            // causes ICE:
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
            // FIXME(#6050): overloaded operators force moves with generic types
            // power *= base_num;
            power = power * base_num;
        }
    }

    #[inline]
    fn shl_unit(&self, n_unit: uint) -> BigUint {
        if n_unit == 0 || self.is_zero() { return (*self).clone(); }

        return BigUint::new(vec::from_elem(n_unit, ZERO_BIG_DIGIT)
                            + self.data);
    }

    #[inline]
    fn shl_bits(&self, n_bits: uint) -> BigUint {
        if n_bits == 0 || self.is_zero() { return (*self).clone(); }

        let mut carry = 0;
        let mut shifted = do self.data.iter().map |elem| {
            let (hi, lo) = BigDigit::from_uint(
                (*elem as uint) << n_bits | (carry as uint)
            );
            carry = hi;
            lo
        }.collect::<~[BigDigit]>();
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
        let mut shifted = ~[];
        for elem in self.data.rev_iter() {
            shifted = ~[(*elem >> n_bits) | borrow] + shifted;
            borrow = *elem << (BigDigit::bits - n_bits);
        }
        return BigUint::new(shifted);
    }

    /// Determines the fewest bits necessary to express the `BigUint`.
    pub fn bits(&self) -> uint {
        if self.is_zero() { return 0; }
        let zeros = self.data.last().leading_zeros();
        return self.data.len()*BigDigit::bits - (zeros as uint);
    }
}

#[cfg(target_word_size = "32")]
#[inline]
fn get_radix_base(radix: uint) -> (uint, uint) {
    assert!(1 < radix && radix <= 16);
    match radix {
        2  => (65536, 16),
        3  => (59049, 10),
        4  => (65536, 8),
        5  => (15625, 6),
        6  => (46656, 6),
        7  => (16807, 5),
        8  => (32768, 5),
        9  => (59049, 5),
        10 => (10000, 4),
        11 => (14641, 4),
        12 => (20736, 4),
        13 => (28561, 4),
        14 => (38416, 4),
        15 => (50625, 4),
        16 => (65536, 4),
        _  => fail2!()
    }
}

#[cfg(target_word_size = "64")]
#[inline]
fn get_radix_base(radix: uint) -> (uint, uint) {
    assert!(1 < radix && radix <= 16);
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
        _  => fail2!()
    }
}

/// A Sign is a `BigInt`'s composing element.
#[deriving(Eq, Clone)]
pub enum Sign { Minus, Zero, Plus }

impl Ord for Sign {
    #[inline]
    fn lt(&self, other: &Sign) -> bool {
        match self.cmp(other) { Less => true, _ => false}
    }
}

impl TotalEq for Sign {
    #[inline]
    fn equals(&self, other: &Sign) -> bool { *self == *other }
}
impl TotalOrd for Sign {
    #[inline]
    fn cmp(&self, other: &Sign) -> Ordering {
        match (*self, *other) {
          (Minus, Minus) | (Zero,  Zero) | (Plus, Plus) => Equal,
          (Minus, Zero)  | (Minus, Plus) | (Zero, Plus) => Less,
          _                                             => Greater
        }
    }
}

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
    priv sign: Sign,
    priv data: BigUint
}

impl Eq for BigInt {
    #[inline]
    fn eq(&self, other: &BigInt) -> bool { self.equals(other) }
}

impl TotalEq for BigInt {
    #[inline]
    fn equals(&self, other: &BigInt) -> bool {
        match self.cmp(other) { Equal => true, _ => false }
    }
}

impl Ord for BigInt {
    #[inline]
    fn lt(&self, other: &BigInt) -> bool {
        match self.cmp(other) { Less => true, _ => false}
    }
}

impl TotalOrd for BigInt {
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

impl ToStr for BigInt {
    #[inline]
    fn to_str(&self) -> ~str { self.to_str_radix(10) }
}

impl FromStr for BigInt {
    #[inline]
    fn from_str(s: &str) -> Option<BigInt> {
        FromStrRadix::from_str_radix(s, 10)
    }
}

impl Num for BigInt {}

impl Orderable for BigInt {
    #[inline]
    fn min(&self, other: &BigInt) -> BigInt {
        if self < other { self.clone() } else { other.clone() }
    }

    #[inline]
    fn max(&self, other: &BigInt) -> BigInt {
        if self > other { self.clone() } else { other.clone() }
    }

    #[inline]
    fn clamp(&self, mn: &BigInt, mx: &BigInt) -> BigInt {
        if self > mx { mx.clone() } else
        if self < mn { mn.clone() } else { self.clone() }
    }
}

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
            (Plus, Plus)   => BigInt::from_biguint(Plus,
                                                   self.data + other.data),
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
        return q;
    }
}

impl Rem<BigInt, BigInt> for BigInt {
    #[inline]
    fn rem(&self, other: &BigInt) -> BigInt {
        let (_, r) = self.div_rem(other);
        return r;
    }
}

impl Neg<BigInt> for BigInt {
    #[inline]
    fn neg(&self) -> BigInt {
        BigInt::from_biguint(self.sign.neg(), self.data.clone())
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
            (_,    Zero)   => fail2!(),
            (Plus, Plus)  | (Zero, Plus)  => ( d,  r),
            (Plus, Minus) | (Zero, Minus) => (-d,  r),
            (Minus, Plus)                 => (-d, -r),
            (Minus, Minus)                => ( d, -r)
        }
    }

    #[inline]
    fn div_floor(&self, other: &BigInt) -> BigInt {
        let (d, _) = self.div_mod_floor(other);
        return d;
    }

    #[inline]
    fn mod_floor(&self, other: &BigInt) -> BigInt {
        let (_, m) = self.div_mod_floor(other);
        return m;
    }

    fn div_mod_floor(&self, other: &BigInt) -> (BigInt, BigInt) {
        // m.sign == other.sign
        let (d_ui, m_ui) = self.data.div_rem(&other.data);
        let d = BigInt::from_biguint(Plus, d_ui);
        let m = BigInt::from_biguint(Plus, m_ui);
        match (self.sign, other.sign) {
            (_,    Zero)   => fail2!(),
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

    /**
     * Calculates the Greatest Common Divisor (GCD) of the number and `other`
     *
     * The result is always positive
     */
    #[inline]
    fn gcd(&self, other: &BigInt) -> BigInt {
        BigInt::from_biguint(Plus, self.data.gcd(&other.data))
    }

    /**
     * Calculates the Lowest Common Multiple (LCM) of the number and `other`
     */
    #[inline]
    fn lcm(&self, other: &BigInt) -> BigInt {
        BigInt::from_biguint(Plus, self.data.lcm(&other.data))
    }

    /// Returns `true` if the number can be divided by `other` without leaving a remainder
    #[inline]
    fn is_multiple_of(&self, other: &BigInt) -> bool { self.data.is_multiple_of(&other.data) }

    /// Returns `true` if the number is divisible by `2`
    #[inline]
    fn is_even(&self) -> bool { self.data.is_even() }

    /// Returns `true` if the number is not divisible by `2`
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
                do self.data.to_u64().and_then |n| {
                    let m: u64 = 1 << 63;
                    if n < m {
                        Some(-(n as i64))
                    } else if n == m {
                        Some(i64::min_value)
                    } else {
                        None
                    }
                }
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
            do FromPrimitive::from_u64(n as u64).and_then |n| {
                Some(BigInt::from_biguint(Plus, n))
            }
        } else if n < 0 {
            do FromPrimitive::from_u64(u64::max_value - (n as u64) + 1).and_then |n| {
                Some(BigInt::from_biguint(Minus, n))
            }
        } else {
            Some(Zero::zero())
        }
    }

    #[inline]
    fn from_u64(n: u64) -> Option<BigInt> {
        if n == 0 {
            Some(Zero::zero())
        } else {
            do FromPrimitive::from_u64(n).and_then |n| {
                Some(BigInt::from_biguint(Plus, n))
            }
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
    fn to_str_radix(&self, radix: uint) -> ~str {
        match self.sign {
            Plus  => self.data.to_str_radix(radix),
            Zero  => ~"0",
            Minus => ~"-" + self.data.to_str_radix(radix)
        }
    }
}

impl FromStrRadix for BigInt {
    /// Creates and initializes an BigInt.
    #[inline]
    fn from_str_radix(s: &str, radix: uint) -> Option<BigInt> {
        BigInt::parse_bytes(s.as_bytes(), radix)
    }
}

trait RandBigInt {
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
        let mut data = vec::with_capacity(digits+1);
        for _ in range(0, digits) {
            data.push(self.gen());
        }
        if rem > 0 {
            let final_digit: BigDigit = self.gen();
            data.push(final_digit >> (BigDigit::bits - rem));
        }
        return BigUint::new(data);
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
        return BigInt::from_biguint(sign, biguint);
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
    /// Creates and initializes an BigInt.
    #[inline]
    pub fn new(sign: Sign, v: ~[BigDigit]) -> BigInt {
        BigInt::from_biguint(sign, BigUint::new(v))
    }

    /// Creates and initializes a `BigInt`.
    #[inline]
    pub fn from_biguint(sign: Sign, data: BigUint) -> BigInt {
        if sign == Zero || data.is_zero() {
            return BigInt { sign: Zero, data: Zero::zero() };
        }
        return BigInt { sign: sign, data: data };
    }

    /// Creates and initializes a `BigInt`.
    #[inline]
    pub fn from_slice(sign: Sign, slice: &[BigDigit]) -> BigInt {
        BigInt::from_biguint(sign, BigUint::from_slice(slice))
    }

    /// Creates and initializes a `BigInt`.
    pub fn parse_bytes(buf: &[u8], radix: uint)
        -> Option<BigInt> {
        if buf.is_empty() { return None; }
        let mut sign  = Plus;
        let mut start = 0;
        if buf[0] == ('-' as u8) {
            sign  = Minus;
            start = 1;
        }
        return BigUint::parse_bytes(buf.slice(start, buf.len()), radix)
            .map_move(|bu| BigInt::from_biguint(sign, bu));
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
    use super::*;

    use std::cmp::{Less, Equal, Greater};
    use std::i64;
    use std::num::{Zero, One, FromStrRadix};
    use std::num::{ToPrimitive, FromPrimitive};
    use std::rand::{task_rng};
    use std::str;
    use std::u64;
    use std::vec;

    #[test]
    fn test_from_slice() {
        fn check(slice: &[BigDigit], data: &[BigDigit]) {
            assert!(data == BigUint::from_slice(slice).data);
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
        let data: ~[BigUint] = [ &[], &[1], &[2], &[-1], &[0, 1], &[2, 1], &[1, 1, 1]  ]
            .map(|v| BigUint::from_slice(*v));
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
        fn check(left: ~[BigDigit],
                 right: ~[BigDigit],
                 expected: ~[BigDigit]) {
            assert_eq!(BigUint::new(left) & BigUint::new(right),
                       BigUint::new(expected));
        }
        check(~[], ~[], ~[]);
        check(~[268, 482, 17],
              ~[964, 54],
              ~[260, 34]);
    }

    #[test]
    fn test_bitor() {
        fn check(left: ~[BigDigit],
                 right: ~[BigDigit],
                 expected: ~[BigDigit]) {
            assert_eq!(BigUint::new(left) | BigUint::new(right),
                       BigUint::new(expected));
        }
        check(~[], ~[], ~[]);
        check(~[268, 482, 17],
              ~[964, 54],
              ~[972, 502, 17]);
    }

    #[test]
    fn test_bitxor() {
        fn check(left: ~[BigDigit],
                 right: ~[BigDigit],
                 expected: ~[BigDigit]) {
            assert_eq!(BigUint::new(left) ^ BigUint::new(right),
                       BigUint::new(expected));
        }
        check(~[], ~[], ~[]);
        check(~[268, 482, 17],
              ~[964, 54],
              ~[712, 468, 17]);
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

        check("1" + "0000" + "0000" + "0000" + "0001" + "0000" + "0000" + "0000" + "0001", 3,
              "8" + "0000" + "0000" + "0000" + "0008" + "0000" + "0000" + "0000" + "0008");
        check("1" + "0000" + "0001" + "0000" + "0001", 2,
              "4" + "0000" + "0004" + "0000" + "0004");
        check("1" + "0001" + "0001", 1,
              "2" + "0002" + "0002");

        check(""  + "4000" + "0000" + "0000" + "0000", 3,
              "2" + "0000" + "0000" + "0000" + "0000");
        check(""  + "4000" + "0000", 2,
              "1" + "0000" + "0000");
        check(""  + "4000", 2,
              "1" + "0000");

        check(""  + "4000" + "0000" + "0000" + "0000", 67,
              "2" + "0000" + "0000" + "0000" + "0000" + "0000" + "0000" + "0000" + "0000");
        check(""  + "4000" + "0000", 35,
              "2" + "0000" + "0000" + "0000" + "0000");
        check(""  + "4000", 19,
              "2" + "0000" + "0000");

        check(""  + "fedc" + "ba98" + "7654" + "3210" + "fedc" + "ba98" + "7654" + "3210", 4,
              "f" + "edcb" + "a987" + "6543" + "210f" + "edcb" + "a987" + "6543" + "2100");
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

        check("1" + "0000" + "0000" + "0000" + "0001" + "0000" + "0000" + "0000" + "0001", 3,
              ""  + "2000" + "0000" + "0000" + "0000" + "2000" + "0000" + "0000" + "0000");
        check("1" + "0000" + "0001" + "0000" + "0001", 2,
              ""  + "4000" + "0000" + "4000" + "0000");
        check("1" + "0001" + "0001", 1,
              ""  + "8000" + "8000");

        check("2" + "0000" + "0000" + "0000" + "0001" + "0000" + "0000" + "0000" + "0001", 67,
              ""  + "4000" + "0000" + "0000" + "0000");
        check("2" + "0000" + "0001" + "0000" + "0001", 35,
              ""  + "4000" + "0000");
        check("2" + "0001" + "0001", 19,
              ""  + "4000");

        check("1" + "0000" + "0000" + "0000" + "0000", 1,
              ""  + "8000" + "0000" + "0000" + "0000");
        check("1" + "0000" + "0000", 1,
              ""  + "8000" + "0000");
        check("1" + "0000", 1,
              ""  + "8000");
        check("f" + "edcb" + "a987" + "6543" + "210f" + "edcb" + "a987" + "6543" + "2100", 4,
              ""  + "fedc" + "ba98" + "7654" + "3210" + "fedc" + "ba98" + "7654" + "3210");

        check("888877776666555544443333222211110000", 16,
              "88887777666655554444333322221111");
    }

    #[cfg(target_word_size = "32")]
    #[test]
    fn test_convert_i64() {
        fn check(b1: BigUint, i: i64) {
            let b2: BigUint = FromPrimitive::from_i64(i).unwrap();
            assert!(b1 == b2);
            assert!(b1.to_i64().unwrap() == i);
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);
        check(i64::max_value.to_biguint().unwrap(), i64::max_value);

        check(BigUint::new(~[                   ]), 0);
        check(BigUint::new(~[ 1                 ]), (1 << (0*BigDigit::bits)));
        check(BigUint::new(~[-1                 ]), (1 << (1*BigDigit::bits)) - 1);
        check(BigUint::new(~[ 0,  1             ]), (1 << (1*BigDigit::bits)));
        check(BigUint::new(~[-1, -1             ]), (1 << (2*BigDigit::bits)) - 1);
        check(BigUint::new(~[ 0,  0,  1         ]), (1 << (2*BigDigit::bits)));
        check(BigUint::new(~[-1, -1, -1         ]), (1 << (3*BigDigit::bits)) - 1);
        check(BigUint::new(~[ 0,  0,  0,  1     ]), (1 << (3*BigDigit::bits)));
        check(BigUint::new(~[-1, -1, -1, -1 >> 1]), i64::max_value);

        assert_eq!(i64::min_value.to_biguint(), None);
        assert_eq!(BigUint::new(~[-1, -1, -1, -1    ]).to_i64(), None);
        assert_eq!(BigUint::new(~[ 0,  0,  0,  0,  1]).to_i64(), None);
        assert_eq!(BigUint::new(~[-1, -1, -1, -1, -1]).to_i64(), None);
    }

    #[cfg(target_word_size = "64")]
    #[test]
    fn test_convert_i64() {
        fn check(b1: BigUint, i: i64) {
            let b2: BigUint = FromPrimitive::from_i64(i).unwrap();
            assert!(b1 == b2);
            assert!(b1.to_i64().unwrap() == i);
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);
        check(i64::max_value.to_biguint().unwrap(), i64::max_value);

        check(BigUint::new(~[           ]), 0);
        check(BigUint::new(~[ 1         ]), (1 << (0*BigDigit::bits)));
        check(BigUint::new(~[-1         ]), (1 << (1*BigDigit::bits)) - 1);
        check(BigUint::new(~[ 0,  1     ]), (1 << (1*BigDigit::bits)));
        check(BigUint::new(~[-1, -1 >> 1]), i64::max_value);

        assert_eq!(i64::min_value.to_biguint(), None);
        assert_eq!(BigUint::new(~[-1, -1    ]).to_i64(), None);
        assert_eq!(BigUint::new(~[ 0,  0,  1]).to_i64(), None);
        assert_eq!(BigUint::new(~[-1, -1, -1]).to_i64(), None);
    }

    #[cfg(target_word_size = "32")]
    #[test]
    fn test_convert_u64() {
        fn check(b1: BigUint, u: u64) {
            let b2: BigUint = FromPrimitive::from_u64(u).unwrap();
            assert!(b1 == b2);
            assert!(b1.to_u64().unwrap() == u);
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);
        check(u64::min_value.to_biguint().unwrap(), u64::min_value);
        check(u64::max_value.to_biguint().unwrap(), u64::max_value);

        check(BigUint::new(~[              ]), 0);
        check(BigUint::new(~[ 1            ]), (1 << (0*BigDigit::bits)));
        check(BigUint::new(~[-1            ]), (1 << (1*BigDigit::bits)) - 1);
        check(BigUint::new(~[ 0,  1        ]), (1 << (1*BigDigit::bits)));
        check(BigUint::new(~[-1, -1        ]), (1 << (2*BigDigit::bits)) - 1);
        check(BigUint::new(~[ 0,  0,  1    ]), (1 << (2*BigDigit::bits)));
        check(BigUint::new(~[-1, -1, -1    ]), (1 << (3*BigDigit::bits)) - 1);
        check(BigUint::new(~[ 0,  0,  0,  1]), (1 << (3*BigDigit::bits)));
        check(BigUint::new(~[-1, -1, -1, -1]), u64::max_value);

        assert_eq!(BigUint::new(~[ 0,  0,  0,  0,  1]).to_u64(), None);
        assert_eq!(BigUint::new(~[-1, -1, -1, -1, -1]).to_u64(), None);
    }

    #[cfg(target_word_size = "64")]
    #[test]
    fn test_convert_u64() {
        fn check(b1: BigUint, u: u64) {
            let b2: BigUint = FromPrimitive::from_u64(u).unwrap();
            assert!(b1 == b2);
            assert!(b1.to_u64().unwrap() == u);
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);
        check(u64::min_value.to_biguint().unwrap(), u64::min_value);
        check(u64::max_value.to_biguint().unwrap(), u64::max_value);

        check(BigUint::new(~[      ]), 0);
        check(BigUint::new(~[ 1    ]), (1 << (0*BigDigit::bits)));
        check(BigUint::new(~[-1    ]), (1 << (1*BigDigit::bits)) - 1);
        check(BigUint::new(~[ 0,  1]), (1 << (1*BigDigit::bits)));
        check(BigUint::new(~[-1, -1]), u64::max_value);

        assert_eq!(BigUint::new(~[ 0,  0,  1]).to_u64(), None);
        assert_eq!(BigUint::new(~[-1, -1, -1]).to_u64(), None);
    }

    #[test]
    fn test_convert_to_bigint() {
        fn check(n: BigUint, ans: BigInt) {
            assert_eq!(n.to_bigint().unwrap(), ans);
            assert_eq!(n.to_bigint().unwrap().to_biguint().unwrap(), n);
        }
        check(Zero::zero(), Zero::zero());
        check(BigUint::new(~[1,2,3]),
              BigInt::from_biguint(Plus, BigUint::new(~[1,2,3])));
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
            let (aVec, bVec, cVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);

            assert!(a + b == c);
            assert!(b + a == c);
        }
    }

    #[test]
    fn test_sub() {
        for elm in sum_triples.iter() {
            let (aVec, bVec, cVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);

            assert!(c - a == b);
            assert!(c - b == a);
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
            let (aVec, bVec, cVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);

            assert!(a * b == c);
            assert!(b * a == c);
        }

        for elm in div_rem_quadruples.iter() {
            let (aVec, bVec, cVec, dVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);
            let d = BigUint::from_slice(dVec);

            assert!(a == b * c + d);
            assert!(a == c * b + d);
        }
    }

    #[test]
    fn test_div_rem() {
        for elm in mul_triples.iter() {
            let (aVec, bVec, cVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);

            if !a.is_zero() {
                assert_eq!(c.div_rem(&a), (b.clone(), Zero::zero()));
            }
            if !b.is_zero() {
                assert_eq!(c.div_rem(&b), (a.clone(), Zero::zero()));
            }
        }

        for elm in div_rem_quadruples.iter() {
            let (aVec, bVec, cVec, dVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);
            let d = BigUint::from_slice(dVec);

            if !b.is_zero() { assert!(a.div_rem(&b) == (c, d)); }
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

    fn to_str_pairs() -> ~[ (BigUint, ~[(uint, ~str)]) ] {
        let bits = BigDigit::bits;
        ~[( Zero::zero(), ~[
            (2, ~"0"), (3, ~"0")
        ]), ( BigUint::from_slice([ 0xff ]), ~[
            (2,  ~"11111111"),
            (3,  ~"100110"),
            (4,  ~"3333"),
            (5,  ~"2010"),
            (6,  ~"1103"),
            (7,  ~"513"),
            (8,  ~"377"),
            (9,  ~"313"),
            (10, ~"255"),
            (11, ~"212"),
            (12, ~"193"),
            (13, ~"168"),
            (14, ~"143"),
            (15, ~"120"),
            (16, ~"ff")
        ]), ( BigUint::from_slice([ 0xfff ]), ~[
            (2,  ~"111111111111"),
            (4,  ~"333333"),
            (16, ~"fff")
        ]), ( BigUint::from_slice([ 1, 2 ]), ~[
            (2,
             ~"10" +
             str::from_chars(vec::from_elem(bits - 1, '0')) + "1"),
            (4,
             ~"2" +
             str::from_chars(vec::from_elem(bits / 2 - 1, '0')) + "1"),
            (10, match bits {
                32 => ~"8589934593", 16 => ~"131073", _ => fail2!()
            }),
            (16,
             ~"2" +
             str::from_chars(vec::from_elem(bits / 4 - 1, '0')) + "1")
        ]), ( BigUint::from_slice([ 1, 2, 3 ]), ~[
            (2,
             ~"11" +
             str::from_chars(vec::from_elem(bits - 2, '0')) + "10" +
             str::from_chars(vec::from_elem(bits - 1, '0')) + "1"),
            (4,
             ~"3" +
             str::from_chars(vec::from_elem(bits / 2 - 1, '0')) + "2" +
             str::from_chars(vec::from_elem(bits / 2 - 1, '0')) + "1"),
            (10, match bits {
                32 => ~"55340232229718589441",
                16 => ~"12885032961",
                _ => fail2!()
            }),
            (16, ~"3" +
             str::from_chars(vec::from_elem(bits / 4 - 1, '0')) + "2" +
             str::from_chars(vec::from_elem(bits / 4 - 1, '0')) + "1")
        ]) ]
    }

    #[test]
    fn test_to_str_radix() {
        let r = to_str_pairs();
        for num_pair in r.iter() {
            let &(ref n, ref rs) = num_pair;
            for str_pair in rs.iter() {
                let &(ref radix, ref str) = str_pair;
                assert_eq!(&n.to_str_radix(*radix), str);
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
                assert_eq!(n, &FromStrRadix::from_str_radix(*str, *radix).unwrap());
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
                // FIXME(#6102): Assignment operator for BigInt causes ICE
                // f *= FromPrimitive::from_uint(i);
                f = f * FromPrimitive::from_uint(i).unwrap();
            }
            return f;
        }

        fn check(n: uint, s: &str) {
            let n = factor(n);
            let ans = match FromStrRadix::from_str_radix(s, 10) {
                Some(x) => x, None => fail2!()
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
        assert_eq!(BigUint::new(~[0,0,0,0]).bits(), 0);
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

        do 10.times {
            assert_eq!(rng.gen_bigint_range(&FromPrimitive::from_uint(236).unwrap(),
                                            &FromPrimitive::from_uint(237).unwrap()),
                       FromPrimitive::from_uint(236).unwrap());
        }

        let l = FromPrimitive::from_uint(403469000 + 2352).unwrap();
        let u = FromPrimitive::from_uint(403469000 + 3513).unwrap();
        do 1000.times {
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
    use super::*;

    use std::cmp::{Less, Equal, Greater};
    use std::i64;
    use std::num::{Zero, One, FromStrRadix};
    use std::num::{ToPrimitive, FromPrimitive};
    use std::rand::{task_rng};
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
        let mut nums = ~[];
        for s in vs.rev_iter() {
            nums.push(BigInt::from_slice(Minus, *s));
        }
        nums.push(Zero::zero());
        nums.push_all_move(vs.map(|s| BigInt::from_slice(Plus, *s)));

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
        check(i64::min_value.to_bigint().unwrap(), i64::min_value);
        check(i64::max_value.to_bigint().unwrap(), i64::max_value);

        assert_eq!(
            (i64::max_value as u64 + 1).to_bigint().unwrap().to_i64(),
            None);

        assert_eq!(
            BigInt::from_biguint(Plus,  BigUint::new(~[1, 2, 3, 4, 5])).to_i64(),
            None);

        assert_eq!(
            BigInt::from_biguint(Minus, BigUint::new(~[1, 0, 0, 1<<(BigDigit::bits-1)])).to_i64(),
            None);

        assert_eq!(
            BigInt::from_biguint(Minus, BigUint::new(~[1, 2, 3, 4, 5])).to_i64(),
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
        check(u64::min_value.to_bigint().unwrap(), u64::min_value);
        check(u64::max_value.to_bigint().unwrap(), u64::max_value);

        assert_eq!(
            BigInt::from_biguint(Plus, BigUint::new(~[1, 2, 3, 4, 5])).to_u64(),
            None);

        let max_value: BigUint = FromPrimitive::from_u64(u64::max_value).unwrap();
        assert_eq!(BigInt::from_biguint(Minus, max_value).to_u64(), None);
        assert_eq!(BigInt::from_biguint(Minus, BigUint::new(~[1, 2, 3, 4, 5])).to_u64(), None);
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
            Plus, BigUint::new(~[1,2,3]));
        let negative = -positive;

        check(zero, unsigned_zero);
        check(positive, BigUint::new(~[1,2,3]));

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
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

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
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

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
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

            assert!(a * b == c);
            assert!(b * a == c);

            assert!((-a) * b == -c);
            assert!((-b) * a == -c);
        }

        for elm in div_rem_quadruples.iter() {
            let (aVec, bVec, cVec, dVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);
            let d = BigInt::from_slice(Plus, dVec);

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
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

            if !a.is_zero() { check(&c, &a, &b, &Zero::zero()); }
            if !b.is_zero() { check(&c, &b, &a, &Zero::zero()); }
        }

        for elm in div_rem_quadruples.iter() {
            let (aVec, bVec, cVec, dVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);
            let d = BigInt::from_slice(Plus, dVec);

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
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

            if !a.is_zero() { check(&c, &a, &b, &Zero::zero()); }
            if !b.is_zero() { check(&c, &b, &a, &Zero::zero()); }
        }

        for elm in div_rem_quadruples.iter() {
            let (aVec, bVec, cVec, dVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);
            let d = BigInt::from_slice(Plus, dVec);

            if !b.is_zero() {
                check(&a, &b, &c, &d);
            }
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
            assert!(ans == n.to_str_radix(10));
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
            let ans = ans.map_move(|n| {
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
    }

    #[test]
    fn test_neg() {
        assert!(-BigInt::new(Plus,  ~[1, 1, 1]) ==
            BigInt::new(Minus, ~[1, 1, 1]));
        assert!(-BigInt::new(Minus, ~[1, 1, 1]) ==
            BigInt::new(Plus,  ~[1, 1, 1]));
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

        do 10.times {
            assert_eq!(rng.gen_bigint_range(&FromPrimitive::from_uint(236).unwrap(),
                                            &FromPrimitive::from_uint(237).unwrap()),
                       FromPrimitive::from_uint(236).unwrap());
        }

        fn check(l: BigInt, u: BigInt) {
            let mut rng = task_rng();
            do 1000.times {
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
    use super::*;
    use std::{iter, util};
    use std::num::{FromPrimitive, Zero, One};
    use extra::test::BenchHarness;

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
            f0 = util::replace(&mut f1, f2);
        }
        f0
    }

    #[bench]
    fn factorial_100(bh: &mut BenchHarness) {
        do bh.iter { factorial(100);  }
    }

    #[bench]
    fn fib_100(bh: &mut BenchHarness) {
        do bh.iter { fib(100); }
    }

    #[bench]
    fn to_str(bh: &mut BenchHarness) {
        let fac = factorial(100);
        let fib = fib(100);
        do bh.iter { fac.to_str(); }
        do bh.iter { fib.to_str(); }
    }
}
