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

A Big integer (signed version: BigInt, unsigned version: BigUint).

A BigUint is represented as an array of BigDigits.
A BigInt is a combination of BigUint and Sign.
*/

#[deny(vecs_implicitly_copyable)];
#[deny(deprecated_mutable_fields)];

use core::cmp::{Eq, Ord, TotalEq, TotalOrd, Ordering, Less, Equal, Greater};
use core::num::{IntConvertible, Zero, One, ToStrRadix, FromStrRadix};
use core::*;

/**
A BigDigit is a BigUint's composing element.

A BigDigit is half the size of machine word size.
*/
#[cfg(target_arch = "x86")]
#[cfg(target_arch = "arm")]
#[cfg(target_arch = "mips")]
pub type BigDigit = u16;

/**
A BigDigit is a BigUint's composing element.

A BigDigit is half the size of machine word size.
*/
#[cfg(target_arch = "x86_64")]
pub type BigDigit = u32;

pub mod BigDigit {
    use bigint::BigDigit;

    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "arm")]
    #[cfg(target_arch = "mips")]
    pub static bits: uint = 16;

    #[cfg(target_arch = "x86_64")]
    pub static bits: uint = 32;

    pub static base: uint = 1 << bits;
    priv static hi_mask: uint = (-1 as uint) << bits;
    priv static lo_mask: uint = (-1 as uint) >> bits;

    #[inline(always)]
    priv fn get_hi(n: uint) -> BigDigit { (n >> bits) as BigDigit }
    #[inline(always)]
    priv fn get_lo(n: uint) -> BigDigit { (n & lo_mask) as BigDigit }

    /// Split one machine sized unsigned integer into two BigDigits.
    #[inline(always)]
    pub fn from_uint(n: uint) -> (BigDigit, BigDigit) {
        (get_hi(n), get_lo(n))
    }

    /// Join two BigDigits into one machine sized unsigned integer
    #[inline(always)]
    pub fn to_uint(hi: BigDigit, lo: BigDigit) -> uint {
        (lo as uint) | ((hi as uint) << bits)
    }
}

/**
A big unsigned integer type.

A BigUint-typed value BigUint { data: @[a, b, c] } represents a number
(a + b * BigDigit::base + c * BigDigit::base^2).
*/
pub struct BigUint {
    priv data: ~[BigDigit]
}

impl Eq for BigUint {
    #[inline(always)]
    fn eq(&self, other: &BigUint) -> bool { self.equals(other) }
    #[inline(always)]
    fn ne(&self, other: &BigUint) -> bool { !self.equals(other) }
}

impl TotalEq for BigUint {
    #[inline(always)]
    fn equals(&self, other: &BigUint) -> bool {
        match self.cmp(other) { Equal => true, _ => false }
    }
}

impl Ord for BigUint {
    #[inline(always)]
    fn lt(&self, other: &BigUint) -> bool {
        match self.cmp(other) { Less => true, _ => false}
    }
    #[inline(always)]
    fn le(&self, other: &BigUint) -> bool {
        match self.cmp(other) { Less | Equal => true, _ => false }
    }
    #[inline(always)]
    fn ge(&self, other: &BigUint) -> bool {
        match self.cmp(other) { Greater | Equal => true, _ => false }
    }
    #[inline(always)]
    fn gt(&self, other: &BigUint) -> bool {
        match self.cmp(other) { Greater => true, _ => false }
    }
}

impl TotalOrd for BigUint {
    #[inline(always)]
    fn cmp(&self, other: &BigUint) -> Ordering {
        let s_len = self.data.len(), o_len = other.data.len();
        if s_len < o_len { return Less; }
        if s_len > o_len { return Greater;  }

        for self.data.eachi_reverse |i, elm| {
            match (*elm, other.data[i]) {
                (l, r) if l < r => return Less,
                (l, r) if l > r => return Greater,
                _               => loop
            };
        }
        return Equal;
    }
}

impl ToStr for BigUint {
    #[inline(always)]
    fn to_str(&self) -> ~str { self.to_str_radix(10) }
}

impl from_str::FromStr for BigUint {
    #[inline(always)]
    fn from_str(s: &str) -> Option<BigUint> {
        FromStrRadix::from_str_radix(s, 10)
    }
}

impl Shl<uint, BigUint> for BigUint {
    #[inline(always)]
    fn shl(&self, rhs: &uint) -> BigUint {
        let n_unit = *rhs / BigDigit::bits;
        let n_bits = *rhs % BigDigit::bits;
        return self.shl_unit(n_unit).shl_bits(n_bits);
    }
}

impl Shr<uint, BigUint> for BigUint {
    #[inline(always)]
    fn shr(&self, rhs: &uint) -> BigUint {
        let n_unit = *rhs / BigDigit::bits;
        let n_bits = *rhs % BigDigit::bits;
        return self.shr_unit(n_unit).shr_bits(n_bits);
    }
}

impl Zero for BigUint {
    #[inline(always)]
    fn zero() -> BigUint { BigUint::new(~[]) }

    #[inline(always)]
    fn is_zero(&self) -> bool { self.data.is_empty() }
}

impl One for BigUint {
    #[inline(always)]
    fn one() -> BigUint { BigUint::new(~[1]) }
}

impl Unsigned for BigUint {}

impl Add<BigUint, BigUint> for BigUint {
    #[inline(always)]
    fn add(&self, other: &BigUint) -> BigUint {
        let new_len = uint::max(self.data.len(), other.data.len());

        let mut carry = 0;
        let sum = do vec::from_fn(new_len) |i| {
            let ai = if i < self.data.len()  { self.data[i]  } else { 0 };
            let bi = if i < other.data.len() { other.data[i] } else { 0 };
            let (hi, lo) = BigDigit::from_uint(
                (ai as uint) + (bi as uint) + (carry as uint)
            );
            carry = hi;
            lo
        };
        if carry == 0 { return BigUint::new(sum) };
        return BigUint::new(sum + [carry]);
    }
}

impl Sub<BigUint, BigUint> for BigUint {
    #[inline(always)]
    fn sub(&self, other: &BigUint) -> BigUint {
        let new_len = uint::max(self.data.len(), other.data.len());

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

        assert!(borrow == 0);     // <=> assert!((self >= other));
        return BigUint::new(diff);
    }
}

impl Mul<BigUint, BigUint> for BigUint {
    fn mul(&self, other: &BigUint) -> BigUint {
        if self.is_zero() || other.is_zero() { return Zero::zero(); }

        let s_len = self.data.len(), o_len = other.data.len();
        if s_len == 1 { return mul_digit(other, self.data[0]);  }
        if o_len == 1 { return mul_digit(self,  other.data[0]); }

        // Using Karatsuba multiplication
        // (a1 * base + a0) * (b1 * base + b0)
        // = a1*b1 * base^2 +
        //   (a1*b1 + a0*b0 - (a1-b0)*(b1-a0)) * base +
        //   a0*b0
        let half_len = uint::max(s_len, o_len) / 2;
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

        #[inline(always)]
        fn mul_digit(a: &BigUint, n: BigDigit) -> BigUint {
            if n == 0 { return Zero::zero(); }
            if n == 1 { return copy *a; }

            let mut carry = 0;
            let prod = do vec::map(a.data) |ai| {
                let (hi, lo) = BigDigit::from_uint(
                    (*ai as uint) * (n as uint) + (carry as uint)
                );
                carry = hi;
                lo
            };
            if carry == 0 { return BigUint::new(prod) };
            return BigUint::new(prod + [carry]);
        }

        #[inline(always)]
        fn cut_at(a: &BigUint, n: uint) -> (BigUint, BigUint) {
            let mid = uint::min(a.data.len(), n);
            return (BigUint::from_slice(vec::slice(a.data, mid,
                                                   a.data.len())),
                    BigUint::from_slice(vec::slice(a.data, 0, mid)));
        }

        #[inline(always)]
        fn sub_sign(a: BigUint, b: BigUint) -> (Ordering, BigUint) {
            match a.cmp(&b) {
                Less    => (Less,    b - a),
                Greater => (Greater, a - b),
                _       => (Equal,   Zero::zero())
            }
        }
    }
}

impl Quot<BigUint, BigUint> for BigUint {
    #[inline(always)]
    fn quot(&self, other: &BigUint) -> BigUint {
        let (q, _) = self.quot_rem(other);
        return q;
    }
}

impl Rem<BigUint, BigUint> for BigUint {
    #[inline(always)]
    fn rem(&self, other: &BigUint) -> BigUint {
        let (_, r) = self.quot_rem(other);
        return r;
    }
}

impl Neg<BigUint> for BigUint {
    #[inline(always)]
    fn neg(&self) -> BigUint { fail!() }
}

impl Integer for BigUint {
    #[inline(always)]
    fn div(&self, other: &BigUint) -> BigUint {
        let (d, _) = self.div_mod(other);
        return d;
    }

    #[inline(always)]
    fn modulo(&self, other: &BigUint) -> BigUint {
        let (_, m) = self.div_mod(other);
        return m;
    }

    #[inline(always)]
    fn div_mod(&self, other: &BigUint) -> (BigUint, BigUint) {
        if other.is_zero() { fail!() }
        if self.is_zero() { return (Zero::zero(), Zero::zero()); }
        if *other == One::one() { return (copy *self, Zero::zero()); }

        match self.cmp(other) {
            Less    => return (Zero::zero(), copy *self),
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
        let (d, m) = div_mod_inner(self << shift, other << shift);
        return (d, m >> shift);

        #[inline(always)]
        fn div_mod_inner(a: BigUint, b: BigUint) -> (BigUint, BigUint) {
            let mut m = a;
            let mut d = Zero::zero::<BigUint>();
            let mut n = 1;
            while m >= b {
                let mut (d0, d_unit, b_unit) = div_estimate(&m, &b, n);
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
                    loop;
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

        #[inline(always)]
        fn div_estimate(a: &BigUint, b: &BigUint, n: uint)
            -> (BigUint, BigUint, BigUint) {
            if a.data.len() < n {
                return (Zero::zero(), Zero::zero(), copy *a);
            }

            let an = vec::slice(a.data, a.data.len() - n, a.data.len());
            let bn = *b.data.last();
            let mut d = ~[];
            let mut carry = 0;
            for an.each_reverse |elt| {
                let ai = BigDigit::to_uint(carry, *elt);
                let di = ai / (bn as uint);
                assert!(di < BigDigit::base);
                carry = (ai % (bn as uint)) as BigDigit;
                d = ~[di as BigDigit] + d;
            }

            let shift = (a.data.len() - an.len()) - (b.data.len() - 1);
            if shift == 0 {
                return (BigUint::new(d), One::one(), copy *b);
            }
            return (BigUint::from_slice(d).shl_unit(shift),
                    One::one::<BigUint>().shl_unit(shift),
                    b.shl_unit(shift));
        }
    }

    #[inline(always)]
    fn quot_rem(&self, other: &BigUint) -> (BigUint, BigUint) {
        self.div_mod(other)
    }

    /**
     * Calculates the Greatest Common Divisor (GCD) of the number and `other`
     *
     * The result is always positive
     */
    #[inline(always)]
    fn gcd(&self, other: &BigUint) -> BigUint {
        // Use Euclid's algorithm
        let mut m = copy *self, n = copy *other;
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
    #[inline(always)]
    fn lcm(&self, other: &BigUint) -> BigUint { ((*self * *other) / self.gcd(other)) }

    /// Returns `true` if the number can be divided by `other` without leaving a remainder
    #[inline(always)]
    fn is_multiple_of(&self, other: &BigUint) -> bool { (*self % *other).is_zero() }

    /// Returns `true` if the number is divisible by `2`
    #[inline(always)]
    fn is_even(&self) -> bool {
        // Considering only the last digit.
        if self.data.is_empty() {
            true
        } else {
            self.data.last().is_even()
        }
    }

    /// Returns `true` if the number is not divisible by `2`
    #[inline(always)]
    fn is_odd(&self) -> bool { !self.is_even() }
}

impl IntConvertible for BigUint {
    #[inline(always)]
    fn to_int(&self) -> int {
        uint::min(self.to_uint(), int::max_value as uint) as int
    }

    #[inline(always)]
    fn from_int(n: int) -> BigUint {
        if (n < 0) { Zero::zero() } else { BigUint::from_uint(n as uint) }
    }
}

impl ToStrRadix for BigUint {
    #[inline(always)]
    fn to_str_radix(&self, radix: uint) -> ~str {
        assert!(1 < radix && radix <= 16);
        let (base, max_len) = get_radix_base(radix);
        if base == BigDigit::base {
            return fill_concat(self.data, radix, max_len)
        }
        return fill_concat(convert_base(copy *self, base), radix, max_len);

        #[inline(always)]
        fn convert_base(n: BigUint, base: uint) -> ~[BigDigit] {
            let divider    = BigUint::from_uint(base);
            let mut result = ~[];
            let mut m      = n;
            while m > divider {
                let (d, m0) = m.div_mod(&divider);
                result += [m0.to_uint() as BigDigit];
                m = d;
            }
            if !m.is_zero() {
                result += [m.to_uint() as BigDigit];
            }
            return result;
        }

        #[inline(always)]
        fn fill_concat(v: &[BigDigit], radix: uint, l: uint) -> ~str {
            if v.is_empty() { return ~"0" }
            let s = str::concat(vec::reversed(v).map(|n| {
                let s = uint::to_str_radix(*n as uint, radix);
                str::from_chars(vec::from_elem(l - s.len(), '0')) + s
            }));
            str::trim_left_chars(s, ['0']).to_owned()
        }
    }
}

impl FromStrRadix for BigUint {
    /// Creates and initializes an BigUint.
    #[inline(always)]
    pub fn from_str_radix(s: &str, radix: uint)
        -> Option<BigUint> {
        BigUint::parse_bytes(str::to_bytes(s), radix)
    }
}

impl BigUint {
    /// Creates and initializes an BigUint.
    #[inline(always)]
    pub fn new(v: ~[BigDigit]) -> BigUint {
        // omit trailing zeros
        let new_len = v.rposition(|n| *n != 0).map_default(0, |p| *p + 1);

        if new_len == v.len() { return BigUint { data: v }; }
        let mut v = v;
        v.truncate(new_len);
        return BigUint { data: v };
    }

    /// Creates and initializes an BigUint.
    #[inline(always)]
    pub fn from_uint(n: uint) -> BigUint {
        match BigDigit::from_uint(n) {
            (0,  0)  => Zero::zero(),
            (0,  n0) => BigUint::new(~[n0]),
            (n1, n0) => BigUint::new(~[n0, n1])
        }
    }

    /// Creates and initializes an BigUint.
    #[inline(always)]
    pub fn from_slice(slice: &[BigDigit]) -> BigUint {
        return BigUint::new(vec::from_slice(slice));
    }

    /// Creates and initializes an BigUint.
    #[inline(always)]
    pub fn parse_bytes(buf: &[u8], radix: uint)
        -> Option<BigUint> {
        let (base, unit_len) = get_radix_base(radix);
        let base_num: BigUint = BigUint::from_uint(base);

        let mut end             = buf.len();
        let mut n: BigUint      = Zero::zero();
        let mut power: BigUint  = One::one();
        loop {
            let start = uint::max(end, unit_len) - unit_len;
            match uint::parse_bytes(vec::slice(buf, start, end), radix) {
                // FIXME(#6102): Assignment operator for BigInt causes ICE
                // Some(d) => n += BigUint::from_uint(d) * power,
                Some(d) => n = n + BigUint::from_uint(d) * power,
                None    => return None
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

    #[inline(always)]
    pub fn to_uint(&self) -> uint {
        match self.data.len() {
            0 => 0,
            1 => self.data[0] as uint,
            2 => BigDigit::to_uint(self.data[1], self.data[0]),
            _ => uint::max_value
        }
    }

    #[inline(always)]
    priv fn shl_unit(&self, n_unit: uint) -> BigUint {
        if n_unit == 0 || self.is_zero() { return copy *self; }

        return BigUint::new(vec::from_elem(n_unit, 0) + self.data);
    }

    #[inline(always)]
    priv fn shl_bits(&self, n_bits: uint) -> BigUint {
        if n_bits == 0 || self.is_zero() { return copy *self; }

        let mut carry = 0;
        let shifted = do vec::map(self.data) |elem| {
            let (hi, lo) = BigDigit::from_uint(
                (*elem as uint) << n_bits | (carry as uint)
            );
            carry = hi;
            lo
        };
        if carry == 0 { return BigUint::new(shifted); }
        return BigUint::new(shifted + [carry]);
    }

    #[inline(always)]
    priv fn shr_unit(&self, n_unit: uint) -> BigUint {
        if n_unit == 0 { return copy *self; }
        if self.data.len() < n_unit { return Zero::zero(); }
        return BigUint::from_slice(
            vec::slice(self.data, n_unit, self.data.len())
        );
    }

    #[inline(always)]
    priv fn shr_bits(&self, n_bits: uint) -> BigUint {
        if n_bits == 0 || self.data.is_empty() { return copy *self; }

        let mut borrow = 0;
        let mut shifted = ~[];
        for self.data.each_reverse |elem| {
            shifted = ~[(*elem >> n_bits) | borrow] + shifted;
            borrow = *elem << (BigDigit::bits - n_bits);
        }
        return BigUint::new(shifted);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
priv fn get_radix_base(radix: uint) -> (uint, uint) {
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
        _  => fail!()
    }
}

#[cfg(target_arch = "arm")]
#[cfg(target_arch = "x86")]
#[cfg(target_arch = "mips")]
#[inline(always)]
priv fn get_radix_base(radix: uint) -> (uint, uint) {
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
        _  => fail!()
    }
}

/// A Sign is a BigInt's composing element.
#[deriving(Eq)]
pub enum Sign { Minus, Zero, Plus }

impl Ord for Sign {
    #[inline(always)]
    fn lt(&self, other: &Sign) -> bool {
        match self.cmp(other) { Less => true, _ => false}
    }
    #[inline(always)]
    fn le(&self, other: &Sign) -> bool {
        match self.cmp(other) { Less | Equal => true, _ => false }
    }
    #[inline(always)]
    fn ge(&self, other: &Sign) -> bool {
        match self.cmp(other) { Greater | Equal => true, _ => false }
    }
    #[inline(always)]
    fn gt(&self, other: &Sign) -> bool {
        match self.cmp(other) { Greater => true, _ => false }
    }
}

impl TotalOrd for Sign {
    #[inline(always)]
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
    #[inline(always)]
    fn neg(&self) -> Sign {
        match *self {
          Minus => Plus,
          Zero  => Zero,
          Plus  => Minus
        }
    }
}

/// A big signed integer type.
pub struct BigInt {
    priv sign: Sign,
    priv data: BigUint
}

impl Eq for BigInt {
    #[inline(always)]
    fn eq(&self, other: &BigInt) -> bool { self.equals(other) }
    #[inline(always)]
    fn ne(&self, other: &BigInt) -> bool { !self.equals(other) }
}

impl TotalEq for BigInt {
    #[inline(always)]
    fn equals(&self, other: &BigInt) -> bool {
        match self.cmp(other) { Equal => true, _ => false }
    }
}

impl Ord for BigInt {
    #[inline(always)]
    fn lt(&self, other: &BigInt) -> bool {
        match self.cmp(other) { Less => true, _ => false}
    }
    #[inline(always)]
    fn le(&self, other: &BigInt) -> bool {
        match self.cmp(other) { Less | Equal => true, _ => false }
    }
    #[inline(always)]
    fn ge(&self, other: &BigInt) -> bool {
        match self.cmp(other) { Greater | Equal => true, _ => false }
    }
    #[inline(always)]
    fn gt(&self, other: &BigInt) -> bool {
        match self.cmp(other) { Greater => true, _ => false }
    }
}

impl TotalOrd for BigInt {
    #[inline(always)]
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
    #[inline(always)]
    fn to_str(&self) -> ~str { self.to_str_radix(10) }
}

impl from_str::FromStr for BigInt {
    #[inline(always)]
    fn from_str(s: &str) -> Option<BigInt> {
        FromStrRadix::from_str_radix(s, 10)
    }
}

impl Shl<uint, BigInt> for BigInt {
    #[inline(always)]
    fn shl(&self, rhs: &uint) -> BigInt {
        BigInt::from_biguint(self.sign, self.data << *rhs)
    }
}

impl Shr<uint, BigInt> for BigInt {
    #[inline(always)]
    fn shr(&self, rhs: &uint) -> BigInt {
        BigInt::from_biguint(self.sign, self.data >> *rhs)
    }
}

impl Zero for BigInt {
    #[inline(always)]
    fn zero() -> BigInt {
        BigInt::from_biguint(Zero, Zero::zero())
    }

    #[inline(always)]
    fn is_zero(&self) -> bool { self.sign == Zero }
}

impl One for BigInt {
    #[inline(always)]
    fn one() -> BigInt {
        BigInt::from_biguint(Plus, One::one())
    }
}

impl Signed for BigInt {
    #[inline(always)]
    fn abs(&self) -> BigInt {
        match self.sign {
            Plus | Zero => copy *self,
            Minus => BigInt::from_biguint(Plus, copy self.data)
        }
    }

    #[inline(always)]
    fn signum(&self) -> BigInt {
        match self.sign {
            Plus  => BigInt::from_biguint(Plus, One::one()),
            Minus => BigInt::from_biguint(Minus, One::one()),
            Zero  => Zero::zero(),
        }
    }

    #[inline(always)]
    fn is_positive(&self) -> bool { self.sign == Plus }

    #[inline(always)]
    fn is_negative(&self) -> bool { self.sign == Minus }
}

impl Add<BigInt, BigInt> for BigInt {
    #[inline(always)]
    fn add(&self, other: &BigInt) -> BigInt {
        match (self.sign, other.sign) {
            (Zero, _)      => copy *other,
            (_,    Zero)   => copy *self,
            (Plus, Plus)   => BigInt::from_biguint(Plus,
                                                   self.data + other.data),
            (Plus, Minus)  => self - (-*other),
            (Minus, Plus)  => other - (-*self),
            (Minus, Minus) => -((-self) + (-*other))
        }
    }
}

impl Sub<BigInt, BigInt> for BigInt {
    #[inline(always)]
    fn sub(&self, other: &BigInt) -> BigInt {
        match (self.sign, other.sign) {
            (Zero, _)    => -other,
            (_,    Zero) => copy *self,
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
    #[inline(always)]
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

impl Quot<BigInt, BigInt> for BigInt {
    #[inline(always)]
    fn quot(&self, other: &BigInt) -> BigInt {
        let (q, _) = self.quot_rem(other);
        return q;
    }
}

impl Rem<BigInt, BigInt> for BigInt {
    #[inline(always)]
    fn rem(&self, other: &BigInt) -> BigInt {
        let (_, r) = self.quot_rem(other);
        return r;
    }
}

impl Neg<BigInt> for BigInt {
    #[inline(always)]
    fn neg(&self) -> BigInt {
        BigInt::from_biguint(self.sign.neg(), copy self.data)
    }
}

impl Integer for BigInt {
    #[inline(always)]
    fn div(&self, other: &BigInt) -> BigInt {
        let (d, _) = self.div_mod(other);
        return d;
    }

    #[inline(always)]
    fn modulo(&self, other: &BigInt) -> BigInt {
        let (_, m) = self.div_mod(other);
        return m;
    }

    #[inline(always)]
    fn div_mod(&self, other: &BigInt) -> (BigInt, BigInt) {
        // m.sign == other.sign
        let (d_ui, m_ui) = self.data.quot_rem(&other.data);
        let d = BigInt::from_biguint(Plus, d_ui),
            m = BigInt::from_biguint(Plus, m_ui);
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

    #[inline(always)]
    fn quot_rem(&self, other: &BigInt) -> (BigInt, BigInt) {
        // r.sign == self.sign
        let (q_ui, r_ui) = self.data.div_mod(&other.data);
        let q = BigInt::from_biguint(Plus, q_ui);
        let r = BigInt::from_biguint(Plus, r_ui);
        match (self.sign, other.sign) {
            (_,    Zero)   => fail!(),
            (Plus, Plus)  | (Zero, Plus)  => ( q,  r),
            (Plus, Minus) | (Zero, Minus) => (-q,  r),
            (Minus, Plus)                 => (-q, -r),
            (Minus, Minus)                => ( q, -r)
        }
    }

    /**
     * Calculates the Greatest Common Divisor (GCD) of the number and `other`
     *
     * The result is always positive
     */
    #[inline(always)]
    fn gcd(&self, other: &BigInt) -> BigInt {
        BigInt::from_biguint(Plus, self.data.gcd(&other.data))
    }

    /**
     * Calculates the Lowest Common Multiple (LCM) of the number and `other`
     */
    #[inline(always)]
    fn lcm(&self, other: &BigInt) -> BigInt {
        BigInt::from_biguint(Plus, self.data.lcm(&other.data))
    }

    /// Returns `true` if the number can be divided by `other` without leaving a remainder
    #[inline(always)]
    fn is_multiple_of(&self, other: &BigInt) -> bool { self.data.is_multiple_of(&other.data) }

    /// Returns `true` if the number is divisible by `2`
    #[inline(always)]
    fn is_even(&self) -> bool { self.data.is_even() }

    /// Returns `true` if the number is not divisible by `2`
    #[inline(always)]
    fn is_odd(&self) -> bool { self.data.is_odd() }
}

impl IntConvertible for BigInt {
    #[inline(always)]
    fn to_int(&self) -> int {
        match self.sign {
            Plus  => uint::min(self.to_uint(), int::max_value as uint) as int,
            Zero  => 0,
            Minus => uint::min((-self).to_uint(),
                               (int::max_value as uint) + 1) as int
        }
    }

    #[inline(always)]
    fn from_int(n: int) -> BigInt {
        if n > 0 {
           return BigInt::from_biguint(Plus,  BigUint::from_uint(n as uint));
        }
        if n < 0 {
            return BigInt::from_biguint(
                Minus, BigUint::from_uint(uint::max_value - (n as uint) + 1)
            );
        }
        return Zero::zero();
    }
}

impl ToStrRadix for BigInt {
    #[inline(always)]
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
    #[inline(always)]
    fn from_str_radix(s: &str, radix: uint)
        -> Option<BigInt> {
        BigInt::parse_bytes(str::to_bytes(s), radix)
    }
}

pub impl BigInt {
    /// Creates and initializes an BigInt.
    #[inline(always)]
    pub fn new(sign: Sign, v: ~[BigDigit]) -> BigInt {
        BigInt::from_biguint(sign, BigUint::new(v))
    }

    /// Creates and initializes an BigInt.
    #[inline(always)]
    pub fn from_biguint(sign: Sign, data: BigUint) -> BigInt {
        if sign == Zero || data.is_zero() {
            return BigInt { sign: Zero, data: Zero::zero() };
        }
        return BigInt { sign: sign, data: data };
    }

    /// Creates and initializes an BigInt.
    #[inline(always)]
    pub fn from_uint(n: uint) -> BigInt {
        if n == 0 { return Zero::zero(); }
        return BigInt::from_biguint(Plus, BigUint::from_uint(n));
    }

    /// Creates and initializes an BigInt.
    #[inline(always)]
    pub fn from_slice(sign: Sign, slice: &[BigDigit]) -> BigInt {
        BigInt::from_biguint(sign, BigUint::from_slice(slice))
    }

    /// Creates and initializes an BigInt.
    #[inline(always)]
    pub fn parse_bytes(buf: &[u8], radix: uint)
        -> Option<BigInt> {
        if buf.is_empty() { return None; }
        let mut sign  = Plus;
        let mut start = 0;
        if buf[0] == ('-' as u8) {
            sign  = Minus;
            start = 1;
        }
        return BigUint::parse_bytes(vec::slice(buf, start, buf.len()), radix)
            .map_consume(|bu| BigInt::from_biguint(sign, bu));
    }

    #[inline(always)]
    fn to_uint(&self) -> uint {
        match self.sign {
            Plus  => self.data.to_uint(),
            Zero  => 0,
            Minus => 0
        }
    }
}

#[cfg(test)]
mod biguint_tests {

    use core::*;
    use core::num::{IntConvertible, Zero, One, FromStrRadix};
    use core::cmp::{Less, Equal, Greater};
    use super::{BigUint, BigDigit};

    #[test]
    fn test_from_slice() {
        fn check(slice: &[BigDigit], data: &[BigDigit]) {
            assert!(data == BigUint::from_slice(slice).data);
        }
        check(~[1], ~[1]);
        check(~[0, 0, 0], ~[]);
        check(~[1, 2, 0, 0], ~[1, 2]);
        check(~[0, 0, 1, 2], ~[0, 0, 1, 2]);
        check(~[0, 0, 1, 2, 0, 0], ~[0, 0, 1, 2]);
        check(~[-1], ~[-1]);
    }

    #[test]
    fn test_cmp() {
        let data = [ &[], &[1], &[2], &[-1], &[0, 1], &[2, 1], &[1, 1, 1]  ]
            .map(|v| BigUint::from_slice(*v));
        for data.eachi |i, ni| {
            for vec::slice(data, i, data.len()).eachi |j0, nj| {
                let j = j0 + i;
                if i == j {
                    assert_eq!(ni.cmp(nj), Equal);
                    assert_eq!(nj.cmp(ni), Equal);
                    assert!(ni == nj);
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
    fn test_shl() {
        fn check(v: ~[BigDigit], shift: uint, ans: ~[BigDigit]) {
            assert!(BigUint::new(v) << shift == BigUint::new(ans));
        }

        check(~[], 3, ~[]);
        check(~[1, 1, 1], 3, ~[1 << 3, 1 << 3, 1 << 3]);
        check(~[1 << (BigDigit::bits - 2)], 2, ~[0, 1]);
        check(~[1 << (BigDigit::bits - 2)], 3, ~[0, 2]);
        check(~[1 << (BigDigit::bits - 2)], 3 + BigDigit::bits, ~[0, 0, 2]);

        test_shl_bits();

        #[cfg(target_arch = "x86_64")]
        fn test_shl_bits() {
            check(~[0x7654_3210, 0xfedc_ba98,
                    0x7654_3210, 0xfedc_ba98], 4,
                  ~[0x6543_2100, 0xedcb_a987,
                    0x6543_210f, 0xedcb_a987, 0xf]);
            check(~[0x2222_1111, 0x4444_3333,
                    0x6666_5555, 0x8888_7777], 16,
                  ~[0x1111_0000, 0x3333_2222,
                    0x5555_4444, 0x7777_6666, 0x8888]);
        }

        #[cfg(target_arch = "arm")]
        #[cfg(target_arch = "x86")]
        #[cfg(target_arch = "mips")]
        fn test_shl_bits() {
            check(~[0x3210, 0x7654, 0xba98, 0xfedc,
                    0x3210, 0x7654, 0xba98, 0xfedc], 4,
                  ~[0x2100, 0x6543, 0xa987, 0xedcb,
                    0x210f, 0x6543, 0xa987, 0xedcb, 0xf]);
            check(~[0x1111, 0x2222, 0x3333, 0x4444,
                    0x5555, 0x6666, 0x7777, 0x8888], 16,
                  ~[0x0000, 0x1111, 0x2222, 0x3333,
                    0x4444, 0x5555, 0x6666, 0x7777, 0x8888]);
        }

    }

    #[test]
    #[ignore(cfg(target_arch = "x86"))]
    #[ignore(cfg(target_arch = "arm"))]
    #[ignore(cfg(target_arch = "mips"))]
    fn test_shr() {
        fn check(v: ~[BigDigit], shift: uint, ans: ~[BigDigit]) {
            assert!(BigUint::new(v) >> shift == BigUint::new(ans));
        }

        check(~[], 3, ~[]);
        check(~[1, 1, 1], 3,
              ~[1 << (BigDigit::bits - 3), 1 << (BigDigit::bits - 3)]);
        check(~[1 << 2], 2, ~[1]);
        check(~[1, 2], 3, ~[1 << (BigDigit::bits - 2)]);
        check(~[1, 1, 2], 3 + BigDigit::bits, ~[1 << (BigDigit::bits - 2)]);
        check(~[0, 1], 1, ~[0x80000000]);
        test_shr_bits();

        #[cfg(target_arch = "x86_64")]
        fn test_shr_bits() {
            check(~[0x6543_2100, 0xedcb_a987,
                    0x6543_210f, 0xedcb_a987, 0xf], 4,
                  ~[0x7654_3210, 0xfedc_ba98,
                    0x7654_3210, 0xfedc_ba98]);
            check(~[0x1111_0000, 0x3333_2222,
                    0x5555_4444, 0x7777_6666, 0x8888], 16,
                  ~[0x2222_1111, 0x4444_3333,
                    0x6666_5555, 0x8888_7777]);
        }

        #[cfg(target_arch = "arm")]
        #[cfg(target_arch = "x86")]
        #[cfg(target_arch = "mips")]
        fn test_shr_bits() {
            check(~[0x2100, 0x6543, 0xa987, 0xedcb,
                    0x210f, 0x6543, 0xa987, 0xedcb, 0xf], 4,
                  ~[0x3210, 0x7654, 0xba98, 0xfedc,
                    0x3210, 0x7654, 0xba98, 0xfedc]);
            check(~[0x0000, 0x1111, 0x2222, 0x3333,
                    0x4444, 0x5555, 0x6666, 0x7777, 0x8888], 16,
                  ~[0x1111, 0x2222, 0x3333, 0x4444,
                    0x5555, 0x6666, 0x7777, 0x8888]);
        }
    }

    #[test]
    fn test_convert_int() {
        fn check(v: ~[BigDigit], i: int) {
            let b = BigUint::new(v);
            assert!(b == IntConvertible::from_int(i));
            assert!(b.to_int() == i);
        }

        check(~[], 0);
        check(~[1], 1);
        check(~[-1], (uint::max_value >> BigDigit::bits) as int);
        check(~[ 0,  1], ((uint::max_value >> BigDigit::bits) + 1) as int);
        check(~[-1, -1 >> 1], int::max_value);

        assert!(BigUint::new(~[0, -1]).to_int() == int::max_value);
        assert!(BigUint::new(~[0, 0, 1]).to_int() == int::max_value);
        assert!(BigUint::new(~[0, 0, -1]).to_int() == int::max_value);
    }

    #[test]
    fn test_convert_uint() {
        fn check(v: ~[BigDigit], u: uint) {
            let b = BigUint::new(v);
            assert!(b == BigUint::from_uint(u));
            assert!(b.to_uint() == u);
        }

        check(~[], 0);
        check(~[ 1], 1);
        check(~[-1], uint::max_value >> BigDigit::bits);
        check(~[ 0,  1], (uint::max_value >> BigDigit::bits) + 1);
        check(~[ 0, -1], uint::max_value << BigDigit::bits);
        check(~[-1, -1], uint::max_value);

        assert!(BigUint::new(~[0, 0, 1]).to_uint()  == uint::max_value);
        assert!(BigUint::new(~[0, 0, -1]).to_uint() == uint::max_value);
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
        for sum_triples.each |elm| {
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
        for sum_triples.each |elm| {
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

    static quot_rem_quadruples: &'static [(&'static [BigDigit],
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
        for mul_triples.each |elm| {
            let (aVec, bVec, cVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);

            assert!(a * b == c);
            assert!(b * a == c);
        }

        for quot_rem_quadruples.each |elm| {
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
    fn test_quot_rem() {
        for mul_triples.each |elm| {
            let (aVec, bVec, cVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);

            if !a.is_zero() {
                assert!(c.quot_rem(&a) == (copy b, Zero::zero()));
            }
            if !b.is_zero() {
                assert!(c.quot_rem(&b) == (copy a, Zero::zero()));
            }
        }

        for quot_rem_quadruples.each |elm| {
            let (aVec, bVec, cVec, dVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);
            let d = BigUint::from_slice(dVec);

            if !b.is_zero() { assert!(a.quot_rem(&b) == (c, d)); }
        }
    }

    #[test]
    fn test_gcd() {
        fn check(a: uint, b: uint, c: uint) {
            let big_a = BigUint::from_uint(a);
            let big_b = BigUint::from_uint(b);
            let big_c = BigUint::from_uint(c);

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
            let big_a = BigUint::from_uint(a);
            let big_b = BigUint::from_uint(b);
            let big_c = BigUint::from_uint(c);

            assert_eq!(big_a.lcm(&big_b), big_c);
        }

        check(1, 0, 0);
        check(0, 1, 0);
        check(1, 1, 1);
        check(8, 9, 72);
        check(11, 5, 55);
        check(99, 17, 1683);
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
                32 => ~"8589934593", 16 => ~"131073", _ => fail!()
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
                _ => fail!()
            }),
            (16, ~"3" +
             str::from_chars(vec::from_elem(bits / 4 - 1, '0')) + "2" +
             str::from_chars(vec::from_elem(bits / 4 - 1, '0')) + "1")
        ]) ]
    }

    #[test]
    fn test_to_str_radix() {
        for to_str_pairs().each |num_pair| {
            let &(n, rs) = num_pair;
            for rs.each |str_pair| {
                let &(radix, str) = str_pair;
                assert!(n.to_str_radix(radix) == str);
            }
        }
    }

    #[test]
    fn test_from_str_radix() {
        for to_str_pairs().each |num_pair| {
            let &(n, rs) = num_pair;
            for rs.each |str_pair| {
                let &(radix, str) = str_pair;
                assert_eq!(&n, &FromStrRadix::from_str_radix(str, radix).get());
            }
        }

        assert_eq!(FromStrRadix::from_str_radix::<BigUint>(~"Z", 10), None);
        assert_eq!(FromStrRadix::from_str_radix::<BigUint>(~"_", 2), None);
        assert_eq!(FromStrRadix::from_str_radix::<BigUint>(~"-1", 10), None);
    }

    #[test]
    fn test_factor() {
        fn factor(n: uint) -> BigUint {
            let mut f= One::one::<BigUint>();
            for uint::range(2, n + 1) |i| {
                // FIXME(#6102): Assignment operator for BigInt causes ICE
                // f *= BigUint::from_uint(i);
                f = f * BigUint::from_uint(i);
            }
            return f;
        }

        fn check(n: uint, s: &str) {
            let n = factor(n);
            let ans = match FromStrRadix::from_str_radix(s, 10) {
                Some(x) => x, None => fail!()
            };
            assert!(n == ans);
        }

        check(3, "6");
        check(10, "3628800");
        check(20, "2432902008176640000");
        check(30, "265252859812191058636308480000000");
    }
}

#[cfg(test)]
mod bigint_tests {
    use super::{BigInt, BigUint, BigDigit, Sign, Minus, Zero, Plus};
    use core::*;
    use core::cmp::{Less, Equal, Greater};
    use core::num::{IntConvertible, Zero, One, FromStrRadix};

    #[test]
    fn test_from_biguint() {
        fn check(inp_s: Sign, inp_n: uint, ans_s: Sign, ans_n: uint) {
            let inp = BigInt::from_biguint(inp_s, BigUint::from_uint(inp_n));
            let ans = BigInt { sign: ans_s, data: BigUint::from_uint(ans_n)};
            assert!(inp == ans);
        }
        check(Plus, 1, Plus, 1);
        check(Plus, 0, Zero, 0);
        check(Minus, 1, Minus, 1);
        check(Zero, 1, Zero, 0);
    }

    #[test]
    fn test_cmp() {
        let vs = [ &[2], &[1, 1], &[2, 1], &[1, 1, 1] ];
        let mut nums = vec::reversed(vs)
            .map(|s| BigInt::from_slice(Minus, *s));
        nums.push(Zero::zero());
        nums.push_all_move(vs.map(|s| BigInt::from_slice(Plus, *s)));

        for nums.eachi |i, ni| {
            for vec::slice(nums, i, nums.len()).eachi |j0, nj| {
                let j = i + j0;
                if i == j {
                    assert_eq!(ni.cmp(nj), Equal);
                    assert_eq!(nj.cmp(ni), Equal);
                    assert!(ni == nj);
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
    fn test_convert_int() {
        fn check(b: BigInt, i: int) {
            assert!(b == IntConvertible::from_int(i));
            assert!(b.to_int() == i);
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);
        check(BigInt::from_biguint(
            Plus, BigUint::from_uint(int::max_value as uint)
        ), int::max_value);

        assert!(BigInt::from_biguint(
            Plus, BigUint::from_uint(int::max_value as uint + 1)
        ).to_int() == int::max_value);
        assert!(BigInt::from_biguint(
            Plus, BigUint::new(~[1, 2, 3])
        ).to_int() == int::max_value);

        check(BigInt::from_biguint(
            Minus, BigUint::from_uint(-int::min_value as uint)
        ), int::min_value);
        assert!(BigInt::from_biguint(
            Minus, BigUint::from_uint(-int::min_value as uint + 1)
        ).to_int() == int::min_value);
        assert!(BigInt::from_biguint(
            Minus, BigUint::new(~[1, 2, 3])
        ).to_int() == int::min_value);
    }

    #[test]
    fn test_convert_uint() {
        fn check(b: BigInt, u: uint) {
            assert!(b == BigInt::from_uint(u));
            assert!(b.to_uint() == u);
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);

        check(
            BigInt::from_biguint(Plus, BigUint::from_uint(uint::max_value)),
            uint::max_value);
        assert!(BigInt::from_biguint(
            Plus, BigUint::new(~[1, 2, 3])
        ).to_uint() == uint::max_value);

        assert!(BigInt::from_biguint(
            Minus, BigUint::from_uint(uint::max_value)
        ).to_uint() == 0);
        assert!(BigInt::from_biguint(
            Minus, BigUint::new(~[1, 2, 3])
        ).to_uint() == 0);
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
        for sum_triples.each |elm| {
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
            assert!((-a) + (-b) == (-c));
            assert!(a + (-a) == Zero::zero());
        }
    }

    #[test]
    fn test_sub() {
        for sum_triples.each |elm| {
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

            assert!(c - a == b);
            assert!(c - b == a);
            assert!((-b) - a == (-c));
            assert!((-a) - b == (-c));
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

    static quot_rem_quadruples: &'static [(&'static [BigDigit],
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
        for mul_triples.each |elm| {
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

            assert!(a * b == c);
            assert!(b * a == c);

            assert!((-a) * b == -c);
            assert!((-b) * a == -c);
        }

        for quot_rem_quadruples.each |elm| {
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
    fn test_div_mod() {
        fn check_sub(a: &BigInt, b: &BigInt, ans_d: &BigInt, ans_m: &BigInt) {
            let (d, m) = a.div_mod(b);
            if !m.is_zero() {
                assert!(m.sign == b.sign);
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

        for mul_triples.each |elm| {
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

            if !a.is_zero() { check(&c, &a, &b, &Zero::zero()); }
            if !b.is_zero() { check(&c, &b, &a, &Zero::zero()); }
        }

        for quot_rem_quadruples.each |elm| {
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
    fn test_quot_rem() {
        fn check_sub(a: &BigInt, b: &BigInt, ans_q: &BigInt, ans_r: &BigInt) {
            let (q, r) = a.quot_rem(b);
            if !r.is_zero() {
                assert!(r.sign == a.sign);
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
        for mul_triples.each |elm| {
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

            if !a.is_zero() { check(&c, &a, &b, &Zero::zero()); }
            if !b.is_zero() { check(&c, &b, &a, &Zero::zero()); }
        }

        for quot_rem_quadruples.each |elm| {
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
            let big_a: BigInt = IntConvertible::from_int(a);
            let big_b: BigInt = IntConvertible::from_int(b);
            let big_c: BigInt = IntConvertible::from_int(c);

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
            let big_a: BigInt = IntConvertible::from_int(a);
            let big_b: BigInt = IntConvertible::from_int(b);
            let big_c: BigInt = IntConvertible::from_int(c);

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
    fn test_to_str_radix() {
        fn check(n: int, ans: &str) {
            assert!(ans == IntConvertible::from_int::<BigInt>(n).to_str_radix(10));
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
            let ans = ans.map(|&n| IntConvertible::from_int::<BigInt>(n));
            assert!(FromStrRadix::from_str_radix(s, 10) == ans);
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
        assert!(-Zero::zero::<BigInt>() == Zero::zero::<BigInt>());
    }
}

