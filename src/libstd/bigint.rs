/*!

A Big integer (signed version: BigInt, unsigned version: BigUint).

A BigUint is represented as an array of BigDigits.
A BigInt is a combination of BigUint and Sign.
*/

use core::cmp::{Eq, Ord};
use core::num::{Num, Zero, One};
use core::*;

/**
A BigDigit is a BigUint's composing element.

A BigDigit is half the size of machine word size.
*/
#[cfg(target_arch = "x86")]
#[cfg(target_arch = "arm")]
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
    pub const bits: uint = 16;

    #[cfg(target_arch = "x86_64")]
    pub const bits: uint = 32;

    pub const max_value: BigDigit = !0;
    pub const min_value: BigDigit =  0;

    pub const base: uint = 1 << bits;
    priv const hi_mask: uint = (-1 as uint) << bits;
    priv const lo_mask: uint = (-1 as uint) >> bits;

    priv pure fn get_hi(n: uint) -> BigDigit { (n >> bits) as BigDigit }
    priv pure fn get_lo(n: uint) -> BigDigit { (n & lo_mask) as BigDigit }

    /// Split one machine sized unsigned integer into two BigDigits.
    #[inline(always)]
    pub pure fn from_uint(n: uint) -> (BigDigit, BigDigit) {
        (get_hi(n), get_lo(n))
    }

    /// Join two BigDigits into one machine sized unsigned integer
    #[inline(always)]
    pub pure fn to_uint(hi: BigDigit, lo: BigDigit) -> uint {
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

impl BigUint : Eq {
    #[inline(always)]
    pure fn eq(&self, other: &BigUint) -> bool { self.cmp(other) == 0 }
    #[inline(always)]
    pure fn ne(&self, other: &BigUint) -> bool { self.cmp(other) != 0 }
}

impl BigUint : Ord {
    #[inline(always)]
    pure fn lt(&self, other: &BigUint) -> bool { self.cmp(other) <  0 }
    #[inline(always)]
    pure fn le(&self, other: &BigUint) -> bool { self.cmp(other) <= 0 }
    #[inline(always)]
    pure fn ge(&self, other: &BigUint) -> bool { self.cmp(other) >= 0 }
    #[inline(always)]
    pure fn gt(&self, other: &BigUint) -> bool { self.cmp(other) >  0 }
}

impl BigUint : ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { self.to_str_radix(10) }
}

impl BigUint : from_str::FromStr {
    #[inline(always)]
    static pure fn from_str(s: &str) -> Option<BigUint> {
        BigUint::from_str_radix(s, 10)
    }
}

impl BigUint : Shl<uint, BigUint> {
    #[inline(always)]
    pure fn shl(&self, rhs: &uint) -> BigUint {
        let mut shifted = ~[];
        unsafe { // call impure function
            vec_ops::shl_set(&mut shifted, self.data, *rhs);
        }
        return BigUint::new(shifted);
    }
}

impl BigUint : Shr<uint, BigUint> {
    #[inline(always)]
    pure fn shr(&self, rhs: &uint) -> BigUint {
        let mut shifted = ~[];
        unsafe { // call impure function
            vec_ops::shr_set(&mut shifted, self.data, *rhs);
        }
        return BigUint::new(shifted);
    }
}

impl BigUint : Zero {
    #[inline(always)]
    static pure fn zero() -> BigUint { BigUint::new(~[]) }
}

impl BigUint : One {
    #[inline(always)]
    static pub pure fn one() -> BigUint { BigUint::new(~[1]) }
}

impl BigUint : Num {
    #[inline(always)]
    pure fn add(&self, other: &BigUint) -> BigUint {
        let mut sum = ~[];
        unsafe { // call impure function
            vec_ops::add_set(&mut sum, self.data, other.data);
        }
        return BigUint::new(sum);
    }

    #[inline(always)]
    pure fn sub(&self, other: &BigUint) -> BigUint {
        let mut diff = ~[];
        unsafe { // call impure function
            vec_ops::sub_set(&mut diff, self.data, other.data);
        }
        return BigUint::new(diff);
    }

    #[inline(always)]
    pure fn mul(&self, other: &BigUint) -> BigUint {
        let mut prod = ~[];
        unsafe { // call impure function
            vec_ops::mul_set(&mut prod, self.data, other.data);
        }
        return BigUint::new(prod);
    }

    #[inline(always)]
    pure fn div(&self, other: &BigUint) -> BigUint {
        let (d, _) = self.divmod(other);
        return d;
    }
    #[inline(always)]
    pure fn modulo(&self, other: &BigUint) -> BigUint {
        let (_, m) = self.divmod(other);
        return m;
    }

    #[inline(always)]
    pure fn neg(&self) -> BigUint { fail }

    #[inline(always)]
    pure fn to_int(&self) -> int {
        uint::min(self.to_uint(), int::max_value as uint) as int
    }

    #[inline(always)]
    static pure fn from_int(n: int) -> BigUint {
        if (n < 0) { Zero::zero() } else { BigUint::from_uint(n as uint) }
    }
}

pub impl BigUint {
    /// Creates and initializes an BigUint.
    #[inline(always)]
    static pub pure fn new(v: ~[BigDigit]) -> BigUint {
        let mut v = v;
        vec_ops::reduce_zeros(&mut v);
        return BigUint { data: v };
    }

    /// Creates and initializes an BigUint.
    #[inline(always)]
    static pub pure fn from_uint(n: uint) -> BigUint {
        let mut buf = ~[];
        unsafe { // call impure function
            vec_ops::from_uint_set(&mut buf, n);
        }
        return BigUint::new(buf);
    }

    /// Creates and initializes an BigUint.
    #[inline(always)]
    static pub pure fn from_slice(slice: &[BigDigit]) -> BigUint {
        return BigUint::new(vec::from_slice(slice));
    }

    /// Creates and initializes an BigUint.
    #[inline(always)]
    static pub pure fn from_str_radix(s: &str, radix: uint)
        -> Option<BigUint> {
        BigUint::parse_bytes(str::to_bytes(s), radix)
    }

    /// Creates and initializes an BigUint.
    #[inline(always)]
    static pub pure fn parse_bytes(buf: &[u8], radix: uint)
        -> Option<BigUint> {
        let result = unsafe { // call impure function
            vec_ops::parse_bytes(buf, radix)
        };
        do option::map_consume(result) |v| {
            BigUint::new(v)
        }
    }

    #[inline(always)]
    pure fn abs(&self) -> BigUint { copy *self }

    /// Compare two BigUint value.
    #[inline(always)]
    pure fn cmp(&self, other: &BigUint) -> int {
        vec_ops::cmp_offset(self.data, other.data, 0)
    }

    #[inline(always)]
    pure fn add_assign(&mut self, other: &BigUint) {
        unsafe { // call impure function
            vec_ops::add_offset_assign(&mut self.data, other.data, 0);
        }
    }

    #[inline(always)]
    pure fn sub_assign(&mut self, other: &BigUint) {
        unsafe { // call impure function
            vec_ops::sub_offset_assign(&mut self.data, other.data, 0);
        }
    }

    #[inline(always)]
    pure fn mul_digit(&self, n: BigDigit) -> BigUint {
        let mut prod = ~[];
        unsafe { // call impure function
            vec_ops::mul_digit_set(&mut prod, self.data, n);
        }
        return BigUint::new(prod);
    }

    #[inline(always)]
    pure fn divmod(&self, other: &BigUint) -> (BigUint, BigUint) {
        let mut d = ~[];
        let mut m = ~[];
        unsafe { // call impure function
            vec_ops::divmod_set(&mut d, &mut m, self.data, other.data);
        }
        return (BigUint::new(d), BigUint::new(m));
    }

    #[inline(always)]
    pure fn quot(&self, other: &BigUint) -> BigUint {
        let (q, _) = self.quotrem(other);
        return q;
    }
    #[inline(always)]
    pure fn rem(&self, other: &BigUint) -> BigUint {
        let (_, r) = self.quotrem(other);
        return r;
    }
    #[inline(always)]
    pure fn quotrem(&self, other: &BigUint) -> (BigUint, BigUint) {
        self.divmod(other)
    }

    #[inline(always)]
    pure fn is_zero(&self) -> bool { self.data.is_empty() }
    #[inline(always)]
    pure fn is_not_zero(&self) -> bool { self.data.is_not_empty() }
    #[inline(always)]
    pure fn is_positive(&self) -> bool { self.is_not_zero() }
    #[inline(always)]
    pure fn is_negative(&self) -> bool { false }
    #[inline(always)]
    pure fn is_nonpositive(&self) -> bool { self.is_zero() }
    #[inline(always)]
    pure fn is_nonnegative(&self) -> bool { true }

    #[inline(always)]
    pure fn to_uint(&self) -> uint {
        match self.data.len() {
            0 => 0,
            1 => self.data[0] as uint,
            2 => BigDigit::to_uint(self.data[1], self.data[0]),
            _ => uint::max_value
        }
    }

    #[inline(always)]
    pure fn to_str_radix(&self, radix: uint) -> ~str {
        unsafe { // call impure function
            vec_ops::to_str_radix(self.data, radix)
        }
    }
}

/// A Sign is a BigInt's composing element.
pub enum Sign { Minus, Zero, Plus }

impl Sign : Eq {
    #[inline(always)]
    pure fn eq(&self, other: &Sign) -> bool { self.cmp(other) == 0 }
    #[inline(always)]
    pure fn ne(&self, other: &Sign) -> bool { self.cmp(other) != 0 }
}

impl Sign : Ord {
    #[inline(always)]
    pure fn lt(&self, other: &Sign) -> bool { self.cmp(other) <  0 }
    #[inline(always)]
    pure fn le(&self, other: &Sign) -> bool { self.cmp(other) <= 0 }
    #[inline(always)]
    pure fn ge(&self, other: &Sign) -> bool { self.cmp(other) >= 0 }
    #[inline(always)]
    pure fn gt(&self, other: &Sign) -> bool { self.cmp(other) >  0 }
}

pub impl Sign {
    /// Compare two Sign.
    #[inline(always)]
    pure fn cmp(&self, other: &Sign) -> int {
        match (*self, *other) {
            (Minus, Minus) | (Zero,  Zero) | (Plus, Plus) =>  0,
            (Minus, Zero)  | (Minus, Plus) | (Zero, Plus) => -1,
            _                                             =>  1
        }
    }

    /// Negate Sign value.
    #[inline(always)]
    pure fn neg(&self) -> Sign {
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

impl BigInt : Eq {
    #[inline(always)]
    pure fn eq(&self, other: &BigInt) -> bool { self.cmp(other) == 0 }
    #[inline(always)]
    pure fn ne(&self, other: &BigInt) -> bool { self.cmp(other) != 0 }
}

impl BigInt : Ord {
    #[inline(always)]
    pure fn lt(&self, other: &BigInt) -> bool { self.cmp(other) <  0 }
    #[inline(always)]
    pure fn le(&self, other: &BigInt) -> bool { self.cmp(other) <= 0 }
    #[inline(always)]
    pure fn ge(&self, other: &BigInt) -> bool { self.cmp(other) >= 0 }
    #[inline(always)]
    pure fn gt(&self, other: &BigInt) -> bool { self.cmp(other) >  0 }
}

impl BigInt : ToStr {
    #[inline(always)]
    pure fn to_str() -> ~str { self.to_str_radix(10) }
}

impl BigInt : from_str::FromStr {
    #[inline(always)]
    static pure fn from_str(s: &str) -> Option<BigInt> {
        BigInt::from_str_radix(s, 10)
    }
}

impl BigInt : Shl<uint, BigInt> {
    #[inline(always)]
    pure fn shl(&self, rhs: &uint) -> BigInt {
        BigInt::from_biguint(self.sign, self.data << *rhs)
    }
}

impl BigInt : Shr<uint, BigInt> {
    #[inline(always)]
    pure fn shr(&self, rhs: &uint) -> BigInt {
        BigInt::from_biguint(self.sign, self.data >> *rhs)
    }
}

impl BigInt : Zero {
    #[inline(always)]
    static pub pure fn zero() -> BigInt {
        BigInt::from_biguint(Zero, Zero::zero())
    }
}

impl BigInt : One {
    #[inline(always)]
    static pub pure fn one() -> BigInt {
        BigInt::from_biguint(Plus, One::one())
    }
}

impl BigInt : Num {
    #[inline(always)]
    pure fn add(&self, other: &BigInt) -> BigInt {
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
    #[inline(always)]
    pure fn sub(&self, other: &BigInt) -> BigInt {
        match (self.sign, other.sign) {
            (Zero, _)    => -other,
            (_,    Zero) => copy *self,
            (Plus, Plus) => match self.data.cmp(&other.data) {
                s if s < 0 =>
                    BigInt::from_biguint(Minus, other.data - self.data),
                s if s > 0 =>
                    BigInt::from_biguint(Plus, self.data - other.data),
                _ =>
                    Zero::zero()
            },
            (Plus, Minus) => self + (-*other),
            (Minus, Plus) => -((-self) + *other),
            (Minus, Minus) => (-other) - (-*self)
        }
    }
    #[inline(always)]
    pure fn mul(&self, other: &BigInt) -> BigInt {
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
    #[inline(always)]
    pure fn div(&self, other: &BigInt) -> BigInt {
        let (d, _) = self.divmod(other);
        return d;
    }
    #[inline(always)]
    pure fn modulo(&self, other: &BigInt) -> BigInt {
        let (_, m) = self.divmod(other);
        return m;
    }
    #[inline(always)]
    pure fn neg(&self) -> BigInt {
        BigInt::from_biguint(self.sign.neg(), copy self.data)
    }

    #[inline(always)]
    pure fn to_int(&self) -> int {
        match self.sign {
            Plus  => uint::min(self.to_uint(), int::max_value as uint) as int,
            Zero  => 0,
            Minus => uint::min((-self).to_uint(),
                               (int::max_value as uint) + 1) as int
        }
    }

    #[inline(always)]
    static pure fn from_int(n: int) -> BigInt {
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

pub impl BigInt {
    /// Creates and initializes an BigInt.
    #[inline(always)]
    static pub pure fn new(sign: Sign, v: ~[BigDigit]) -> BigInt {
        BigInt::from_biguint(sign, BigUint::new(v))
    }

    /// Creates and initializes an BigInt.
    #[inline(always)]
    static pub pure fn from_biguint(sign: Sign, data: BigUint) -> BigInt {
        if sign == Zero || data.is_zero() {
            return BigInt { sign: Zero, data: Zero::zero() };
        }
        return BigInt { sign: sign, data: data };
    }

    /// Creates and initializes an BigInt.
    #[inline(always)]
    static pub pure fn from_uint(n: uint) -> BigInt {
        if n == 0 { return Zero::zero(); }
        return BigInt::from_biguint(Plus, BigUint::from_uint(n));
    }

    /// Creates and initializes an BigInt.
    #[inline(always)]
    static pub pure fn from_slice(sign: Sign, slice: &[BigDigit]) -> BigInt {
        BigInt::from_biguint(sign, BigUint::from_slice(slice))
    }

    /// Creates and initializes an BigInt.
    #[inline(always)]
    static pub pure fn from_str_radix(s: &str, radix: uint)
        -> Option<BigInt> {
        BigInt::parse_bytes(str::to_bytes(s), radix)
    }

    /// Creates and initializes an BigInt.
    #[inline(always)]
    static pub pure fn parse_bytes(buf: &[u8], radix: uint)
        -> Option<BigInt> {
        if buf.is_empty() { return None; }
        let mut sign  = Plus;
        let mut start = 0;
        if buf[0] == ('-' as u8) {
            sign  = Minus;
            start = 1;
        }

        let bu = BigUint::parse_bytes(
            vec::view(buf, start, buf.len()), radix);
        return do option::map_consume(bu) |bu| {
            BigInt::from_biguint(sign, bu)
        };
    }

    #[inline(always)]
    pure fn abs(&self) -> BigInt {
        BigInt::from_biguint(Plus, copy self.data)
    }

    #[inline(always)]
    pure fn cmp(&self, other: &BigInt) -> int {
        let ss = self.sign, os = other.sign;
        if ss < os { return -1; }
        if ss > os { return  1; }

        assert ss == os;
        match ss {
            Zero  => 0,
            Plus  => self.data.cmp(&other.data),
            Minus => self.data.cmp(&other.data).neg(),
        }
    }

    #[inline(always)]
    pure fn divmod(&self, other: &BigInt) -> (BigInt, BigInt) {
        // m.sign == other.sign
        let (d_ui, m_ui) = self.data.divmod(&other.data);
        let d = BigInt::from_biguint(Plus, d_ui);
        let m = BigInt::from_biguint(Plus, m_ui);
        match (self.sign, other.sign) {
            (_,    Zero)   => fail,
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
    pure fn quot(&self, other: &BigInt) -> BigInt {
        let (q, _) = self.quotrem(other);
        return q;
    }
    #[inline(always)]
    pure fn rem(&self, other: &BigInt) -> BigInt {
        let (_, r) = self.quotrem(other);
        return r;
    }

    #[inline(always)]
    pure fn quotrem(&self, other: &BigInt) -> (BigInt, BigInt) {
        // r.sign == self.sign
        let (q_ui, r_ui) = self.data.quotrem(&other.data);
        let q = BigInt::from_biguint(Plus, q_ui);
        let r = BigInt::from_biguint(Plus, r_ui);
        match (self.sign, other.sign) {
            (_,    Zero)   => fail,
            (Plus, Plus)  | (Zero, Plus)  => ( q,  r),
            (Plus, Minus) | (Zero, Minus) => (-q,  r),
            (Minus, Plus)                 => (-q, -r),
            (Minus, Minus)                => ( q, -r)
        }
    }

    #[inline(always)]
    pure fn is_zero(&self) -> bool { self.sign == Zero }
    #[inline(always)]
    pure fn is_not_zero(&self) -> bool { self.sign != Zero }
    #[inline(always)]
    pure fn is_positive(&self) -> bool { self.sign == Plus }
    #[inline(always)]
    pure fn is_negative(&self) -> bool { self.sign == Minus }
    #[inline(always)]
    pure fn is_nonpositive(&self) -> bool { self.sign != Plus }
    #[inline(always)]
    pure fn is_nonnegative(&self) -> bool { self.sign != Minus }

    #[inline(always)]
    pure fn to_uint(&self) -> uint {
        match self.sign {
            Plus  => self.data.to_uint(),
            Zero  => 0,
            Minus => 0
        }
    }

    #[inline(always)]
    pure fn to_str_radix(&self, radix: uint) -> ~str {
        match self.sign {
            Plus  => self.data.to_str_radix(radix),
            Zero  => ~"0",
            Minus => ~"-" + self.data.to_str_radix(radix)
        }
    }
}

priv mod vec_ops {

    use core::*;
    #[cfg(stage1)]
    #[cfg(stage2)]
    use super::{BigDigit};

    pub fn from_uint_set(buf: &mut ~[BigDigit], n: uint) {
        let (hi, lo) = BigDigit::from_uint(n);
        if hi == 0 {
            if lo == 0 { assign(buf, &[]); }
            else { assign(buf, &[lo]); }
        } else {
            assign(buf, &[lo, hi]);
        }
    }

    pub pure fn reduce_zeros(v: &mut ~[BigDigit]) {
        let new_len = v.rposition(|n| *n != 0).map_default(0, |p| *p + 1);
        unsafe { vec::raw::set_len(v, new_len); }
    }

    pub pure fn reduce_zeros_view(v: &a/[BigDigit]) -> &a/[BigDigit] {
        let new_len = v.rposition(|n| *n != 0).map_default(0, |p| *p + 1);
        return vec::view(v, 0, new_len);
    }

    pub pure fn cmp_offset(a: &[BigDigit], b: &[BigDigit], offset: uint)
        -> int {
        let a_len = a.len(), b_len = b.len();
        if a_len < b_len + offset { return -1; }
        if a_len > b_len + offset { return  1; }

        let mut i = a_len;
        while i > offset {
            i -= 1;
            if a[i] < b[i - offset] { return -1; }
            if a[i] > b[i - offset] { return  1;}
        }
        return 0;
    }

    pub fn replace_offset(dst: &mut ~[BigDigit], src: &[BigDigit],
                          offset: uint) {
        let s_len = src.len();
        if s_len == 0 { return; }
        let d_len = dst.len();

        vec::reserve(dst, s_len + offset);
        unsafe { vec::raw::set_len(dst, uint::max(s_len + offset, d_len)); }

        if d_len < offset {
            do vec::as_mut_buf(vec::mut_view(*dst, d_len, offset)) |p, len| {
                unsafe { ptr::memset(p, 0, len); }
            }
        }
        unsafe {
            vec::raw::memcpy(vec::mut_view(*dst, offset, s_len + offset),
                             src, s_len);
        }
    }

    pub fn cut_at(a: &a/[BigDigit], n: uint)
        -> (&a/[BigDigit], &a/[BigDigit]) {
        let a_len = a.len();
        let hi, lo;
        if n < a_len {
            hi = vec::view(a, n, a_len);
            lo = vec::view(a, 0, n);
        } else {
            hi = vec::view(a, 0, 0);
            lo = vec::view(a, 0, a_len);
        }
        return (reduce_zeros_view(hi), reduce_zeros_view(lo));
    }

    pub fn assign(dst: &mut ~[BigDigit], src: &[BigDigit]) {
        replace_offset(dst, src, 0);
        unsafe {
            vec::raw::set_len(dst, src.len());
        }
    }

    priv macro_rules! shl_unit (
        ($dst:ident, $src:ident, $n:ident) => ({
            let s_len = $src.len();
            let new_len = $n + s_len;
            vec::reserve($dst, new_len);
            unsafe {
                vec::raw::set_len($dst, new_len);
                vec::raw::memmove(vec::mut_view(*$dst, $n, s_len + $n),
                                  $src.view(0, s_len), s_len);
                do vec::as_mut_buf(*$dst) |ptr, _len| {
                    ptr::memset(ptr, 0, $n);
                }
            }
        })
    )

    priv macro_rules! shl_bits (
        ($dst:ident, $src:ident, $n:ident) => ({
            let s_len = $src.len();
            vec::reserve($dst, s_len);
            unsafe { vec::raw::set_len($dst, s_len); }
            let mut carry = 0;
            for uint::range(0, s_len) |i| {
                let (hi, lo) = BigDigit::from_uint(
                    ($src[i] as uint) << $n | (carry as uint));
                carry = hi;
                $dst[i] = lo;
            }
            if carry != 0 { $dst.push(carry); }
        })
    )

    pub fn shl_set(dst: &mut ~[BigDigit], a: &[BigDigit], n: uint) {
        let n_unit = n / BigDigit::bits;
        let n_bits = n % BigDigit::bits;
        if n_unit > 0 {
            shl_unit!(dst, a, n_unit);
            if n_bits > 0 {
                shl_bits!(dst, dst, n_bits);
            }
        } else if n_bits > 0 {
            shl_bits!(dst, a, n_bits);
        } else {
            assign(dst, a);
        }
    }

    pub fn shl_assign(a: &mut ~[BigDigit], n: uint) {
        let n_unit = n / BigDigit::bits;
        let n_bits = n % BigDigit::bits;
        if n_unit > 0 { shl_unit!(a, a, n_unit); }
        if n_bits > 0 { shl_bits!(a, a, n_bits); }
    }

    priv macro_rules! shr_unit (
        ($dst:ident, $src:ident, $n:ident) => ({
            let s_len = $src.len();
            if s_len < $n {
                unsafe { vec::raw::set_len($dst, 0); }
            } else {
                let new_len = s_len - $n;
                vec::reserve($dst, new_len);
                unsafe {
                    if $dst.len() < new_len {
                        vec::raw::set_len($dst, new_len);
                    }
                    vec::raw::memmove(*$dst, $src.view($n, s_len), new_len);
                    vec::raw::set_len($dst, new_len);
                }
            }
        })
    )

    priv macro_rules! shr_bits (
        ($dst:ident, $src:ident, $n:ident) => ({
            let s_len = $src.len();
            vec::reserve($dst, s_len);
            unsafe { vec::raw::set_len($dst, s_len); }
            if s_len > 0 {
                let mut borrow = 0;
                let mut i = s_len - 1;
                loop {
                    let elem = $src[i];
                    $dst[i] = (elem >> $n) | borrow;
                    borrow = elem << (uint::bits - $n);
                    if i == 0 { break; }
                    i -= 1;
                }
                reduce_zeros($dst);
            }
        })
    )

    pub fn shr_set(dst: &mut ~[BigDigit], a: &[BigDigit], n: uint) {
        let n_unit = n / BigDigit::bits;
        let n_bits = n % BigDigit::bits;
        if n_unit > 0 {
            shr_unit!(dst, a, n_unit);
            if n_bits > 0 {
                shr_bits!(dst, dst, n_bits);
            }
        } else if n_bits > 0 {
            shr_bits!(dst, a, n_bits);
        } else {
            assign(dst, a)
        }
    }

    pub fn shr_assign(a: &mut ~[BigDigit], n: uint) {
        let n_unit = n / BigDigit::bits;
        let n_bits = n % BigDigit::bits;
        if n_unit > 0 { shr_unit!(a, a, n_unit); }
        if n_bits > 0 { shr_bits!(a, a, n_bits); }
    }

    priv pure fn shr_unit_view(a: &a/[BigDigit], n: uint) -> &a/[BigDigit] {
        if a.len() > n { return vec::view(a, n, a.len()); }
        return a;
    }


    priv macro_rules! add_carry {
        (($carry:expr, $dst:expr) = sum($($elem:expr),+)) => ({
            let (hi, lo) = BigDigit::from_uint( 0 $( + ($elem as uint) )+ );
            $carry = hi;
            $dst   = lo;
        })
    }

    pub fn add_offset_assign(a: &mut ~[BigDigit], b: &[BigDigit],
                             offset: uint) {
        let a_len = a.len();
        let b_len = b.len();
        if b_len == 0 { return; }

        if a_len < offset {
            replace_offset(a, b, offset);
            return;
        }

        let new_len = uint::max(a_len, b_len + offset);
        if vec::capacity(a) < new_len {
            // reallocating only if reallocation must be needed
            vec::reserve(a, new_len + 1);
        }
        unsafe { vec::raw::set_len(a, new_len); }

        let mut carry = 0;
        for uint::range(offset, uint::min(a_len, b_len + offset)) |i| {
            add_carry!(
                (carry, a[i]) = sum(a[i], b[i - offset], carry)
            );
        }

        if a_len < b_len + offset {
            let mut i = a_len;
            while i < b_len + offset {
                if carry == 0 { break; }
                add_carry!(
                    (carry, a[i]) = sum(b[i - offset], carry)
                );
                i += 1;
            }

            replace_offset(a, b.view(i - offset, b_len), i);
        } else {
            for uint::range(b_len + offset, a_len) |i| {
                if carry == 0 { break; }
                add_carry!(
                    (carry, a[i]) = sum(a[i], carry)
                );
            }
        }

        if carry != 0 { a.push(carry); }
    }

    pub fn add_set(sum: &mut ~[BigDigit], a: &[BigDigit], b: &[BigDigit]) {
        let a_len = a.len();
        let b_len = b.len();
        if a_len < b_len {
            add_set(sum, b, a);
            return;
        }
        assert a_len >= b_len;

        if vec::capacity(sum) < a_len {
            // reallocating only if reallocation must be needed
            vec::reserve(sum, a_len + 1);
        }
        unsafe { vec::raw::set_len(sum, a_len); }

        let mut carry = 0;
        for uint::range(0, b_len) |i| {
            add_carry!(
                (carry, sum[i]) = sum(a[i], b[i], carry)
            );
        }
        let mut i = b_len;
        while i < a_len {
            if carry == 0 { break; }
            add_carry!(
                (carry, sum[i]) = sum(a[i], carry)
            );
            i += 1;
        }
        if i < a_len {
            replace_offset(sum, a.view(i, a_len), i);
        }
        if carry != 0 { sum.push(carry); }
    }

    priv macro_rules! sub_borrow {
        (($borrow:expr, $dst:expr) = sub($($elem:expr),+)) => ({
            let (hi, lo) = BigDigit::from_uint(
                BigDigit::base + $( ($elem as uint) )-+ );
            /*
            hi*(base) + lo == 1*(base) + ai - bi - borrow
            =>  ai - bi - borrow < 0 <=> hi == 0
            */
            $borrow = 1 - hi;
            $dst   = lo;
        })
    }

    pub fn sub_offset_assign(a: &mut ~[BigDigit], b: &[BigDigit],
                             offset: uint) {
        if b.is_empty() { return; }
        let c = cmp_offset(*a, b, offset);
        assert c >= 0;
        if c == 0 {
            unsafe { vec::raw::set_len(a, offset); }
            reduce_zeros(a);
            return;
        }

        let a_len = a.len();
        let b_len = b.len();

        let mut borrow = 0;
        for uint::range(offset, b_len + offset) |i| {
            sub_borrow!((borrow, a[i]) = sub(a[i], b[i - offset], borrow));
        }
        for uint::range(b_len + offset, a_len) |i| {
            if borrow == 0 { break; }
            sub_borrow!((borrow, a[i]) = sub(a[i], borrow));
        }
        assert borrow == 0;
        reduce_zeros(a);
    }

    pub fn sub_set(diff: &mut ~[BigDigit], a: &[BigDigit], b: &[BigDigit]) {
        if b.is_empty() {
            replace_offset(diff, a, 0);
            return;
        }
        let c = cmp_offset(a, b, 0);
        assert c >= 0;
        if c == 0 {
            unsafe { vec::raw::set_len(diff, 0); }
            return;
        }

        let a_len = a.len();
        let b_len = b.len();
        vec::reserve(diff, a_len);
        unsafe { vec::raw::set_len(diff, a_len); }

        let mut borrow = 0;
        for uint::range(0, b_len) |i| {
            sub_borrow!((borrow, diff[i]) = sub(a[i], b[i], borrow));
        }
        let mut i = b_len;
        while i < a_len {
            if borrow == 0 { break; }
            sub_borrow!((borrow, diff[i]) = sub(a[i], borrow));
            i += 1;
        }
        if i < a_len {
            replace_offset(diff, a.view(i, a_len), i);
        }
        assert borrow == 0;
        reduce_zeros(diff);
    }

    priv macro_rules! mul_carry {
        (($carry:expr, $dst:expr) = mul($a:expr, $b:expr, $c:expr)) => ({
            let (hi, lo) = BigDigit::from_uint(
                ($a as uint) * ($b as uint) + ($c as uint));
            $carry = hi;
            $dst   = lo;
        })
    }

    pub fn mul_digit_assign(a: &mut ~[BigDigit], n: BigDigit) {
        if n == 0 {
            unsafe { vec::raw::set_len(a, 0); }
            return;
        }
        if n == 1 { return; }

        let mut carry = 0;
        for uint::range(0, a.len()) |i| {
            mul_carry!((carry, a[i]) = mul(a[i], n, carry));
        }
        if carry != 0 { a.push(carry); }
    }


    pub fn mul_digit_set(prod: &mut ~[BigDigit],
                         a: &[BigDigit], n: BigDigit) {
        if n == 0 {
            unsafe { vec::raw::set_len(prod, 0); }
            return;
        }
        if n == 1 {
            assign(prod, a);
            return;
        }

        let a_len = a.len();
        if vec::capacity(prod) < a_len {
            vec::reserve(prod, a_len + 1);
        }
        unsafe { vec::raw::set_len(prod, a_len); }

        let mut carry = 0;
        for uint::range(0, a_len) |i| {
            mul_carry!((carry, prod[i]) = mul(a[i], n, carry));
        }
        if carry != 0 { prod.push(carry); }
    }

    pub fn mul_set(prod: &mut ~[BigDigit],
                   a: &[BigDigit], b: &[BigDigit]) {
        let a_len = a.len();
        let b_len = b.len();
        if a_len == 0 || b_len == 0 {
            unsafe { vec::raw::set_len(prod, 0); }
            return;
        }
        if a_len == 1 { mul_digit_set(prod, b, a[0]); return; }
        if b_len == 1 { mul_digit_set(prod, a, b[0]); return; }

        // Using Karatsuba multiplication
        // (a1 * base + a0) * (b1 * base + b0)
        // = a1*b1 * base^2 +
        //   (a1*b1 + a0*b0 - (a1-b0)*(b1-a0)) * base +
        //   a0*b0
        let half_len = uint::max(a_len, b_len) / 2;
        let (aHi, aLo) = cut_at(a, half_len);
        let (bHi, bLo) = cut_at(b, half_len);

        // XXX: reduce memory allocation
        let mut ll = ~[];
        mul_set(&mut ll, aLo, bLo);
        let mut hh = ~[];
        mul_set(&mut hh, aHi, bHi);
        let mut mm = ~[];
        add_offset_assign(&mut mm, hh, 0);
        add_offset_assign(&mut mm, ll, 0);
        {
            let mut m1 = ~[];
            let s1 = sub_sign_set(&mut m1, aHi, aLo);
            let mut m2 = ~[];
            let s2 = sub_sign_set(&mut m2, bHi, bLo);
            let mut mprod = ~[];
            mul_set(&mut mprod, m1, m2);
            if s1 * s2 < 0 {
                add_offset_assign(&mut mm, mprod, 0);
            } else if s1 * s2 > 0 {
                sub_offset_assign(&mut mm, mprod, 0);
            } else { /* Do nothing */ }
        }

        unsafe {
            add_offset_assign(prod, ll, 0);
            add_offset_assign(prod, mm, half_len);
            add_offset_assign(prod, hh, half_len * 2);
        }

        fn sub_sign_set(diff: &mut ~[BigDigit],
                        a: &[BigDigit], b: &[BigDigit]) -> int {
            let s = cmp_offset(a, b, 0);
            match s {
                s if s < 0 => sub_set(diff, b, a),
                s if s > 0 => sub_set(diff, a, b),
                _          => unsafe { vec::raw::set_len(diff, 0); }
            }
            return s;
        }
    }

    pub fn divmod_set(d: &mut ~[BigDigit], m: &mut ~[BigDigit],
                      a: &[BigDigit], b: &[BigDigit]) {
        let a_len = a.len();
        let b_len = b.len();
        if b_len == 0 { fail }
        if a_len == 0 {
            unsafe { vec::raw::set_len(d, 0); }
            unsafe { vec::raw::set_len(m, 0); }
            return;
        }
        if b_len == 1 && b[0] == 1 {
            assign(d, a);
            unsafe { vec::raw::set_len(m, 0); }
            return;
        }

        let c = cmp_offset(a, b, 0);
        if c < 0 {
            unsafe { vec::raw::set_len(d, 0); }
            assign(m, a);
            return;
        }
        if c == 0 {
            assign(d, &[1]);
            unsafe { vec::raw::set_len(m, 0); }
            return;
        }

        let shift = {
            let mut n = b.last();
            let mut s = 0;
            while n < (1 << BigDigit::bits - 1) {
                n <<= 1;
                s += 1;
            }
            assert s < BigDigit::bits;
            s
        };
        let mut a2 = ~[], b2 = ~[];
        shl_set(&mut a2, a, shift);
        shl_set(&mut b2, b, shift);
        inner(d, m, a2, b2);
        shr_assign(m, shift);

        fn inner(d: &mut ~[BigDigit], m: &mut ~[BigDigit],
                 a: &[BigDigit], b: &[BigDigit]) {
            let b_len = b.len();
            unsafe {
                vec::raw::set_len(d, 0);
                assign(m, a);
            }
            let mut n = 1;
            let mut buf = ~[];
            vec::reserve(&mut buf, b_len + 1);
            while cmp_offset(*m, b, 0) >= 0 {
                let mut d0 = div_estimate(*m, b, n);
                if d0 == 0 {
                    unsafe { vec::raw::set_len(&mut buf, 0); }
                    n = 2;
                    loop;
                }
                mul_digit_set(&mut buf, b, d0);
                let offset = m.len() - b_len - n + 1;
                if cmp_offset(buf, shr_unit_view(*m, offset), 0) > 0 {
                    sub_offset_assign(&mut buf, b, 0);
                    if d0 == 1 {
                        n = 2;
                        loop;
                    }
                    d0 -= 1;
                }
                add_offset_assign(d, &[d0], offset);
                sub_offset_assign(m, buf, offset);
                n = 1;
            }
            reduce_zeros(d);
            reduce_zeros(m);
        }
    }

    priv pure fn div_estimate(a: &[BigDigit], b: &[BigDigit], n: uint)
        -> BigDigit {
        assert n == 1 || n == 2;

        let a_len = a.len();
        if a_len < n { return 0; }

        let b_len = b.len();
        let an = a[a_len - 1];
        let bn = b[b_len - 1];
        if n == 1 { return an / bn; }

        // n == 2
        if an == bn { return -1; }
        let an = BigDigit::to_uint(an, a[a_len - 2]);
        return (an / (bn as uint)) as BigDigit;
    }


    pub fn parse_bytes(buf: &[u8], radix: uint) -> Option<~[BigDigit]> {
        let (base, unit_len) = get_radix_base(radix);
        let buf_len = buf.len();
        let n_len   = uint::div_ceil(buf_len, unit_len);
        let mut n   = ~[];
        vec::reserve(&mut n, n_len);

        if base == BigDigit::base {
            for uint::range(0, n_len) |exp| {
                let end   = buf_len - unit_len * exp;
                let start = uint::max(end, unit_len) - unit_len;
                match uint::parse_bytes(buf.view(start, end), radix) {
                    None    => { return None; }
                    Some(d) => {
                        assert d < BigDigit::base;
                        replace_offset(&mut n, &[d as BigDigit], exp);
                    }
                }
            }
            return Some(n);
        }

        assert base < BigDigit::base;

        let mut end   = buf_len;
        let mut power = ~[1];
        let mut prod  = ~[];
        vec::reserve(&mut power, n_len);
        vec::reserve(&mut prod,  n_len);
        for n_len.times {
            let start = uint::max(end, unit_len) - unit_len;
            match uint::parse_bytes(buf.view(start, end), radix) {
                None    => { return None; }
                Some(d) => {
                    assert d < BigDigit::base;
                    mul_digit_set(&mut prod, power, d as BigDigit);
                    add_offset_assign(&mut n, prod, 0);
                }
            }
            end -= unit_len;
            mul_digit_assign(&mut power, base as BigDigit);
        }
        return Some(n);
    }

    pub fn to_str_radix(buf: &[BigDigit], radix: uint) -> ~str {
        assert 1 < radix && radix <= 16;
        let len = buf.len();
        if len == 0 { return ~"0"; }

        let (base, max_len) = get_radix_base(radix);
        if base == BigDigit::base {
            return fill_concat(buf, radix, max_len)
        }

        let divider = {
            let mut d = ~[];
            from_uint_set(&mut d, base);
            d
        };

        let result_len = len * 2;
        let mut converted = ~[];
        let mut d = ~[];
        let mut m = ~[];
        vec::reserve(&mut converted, result_len);
        vec::reserve(&mut d, result_len);

        assign(&mut d, buf);
        let mut acc = ~[];
        for uint::range(0, result_len) |i| {
            assign(&mut acc, d);
            divmod_set(&mut d, &mut m, acc, divider);
            replace_offset(&mut converted, m, i);
            if d.len() == 0 { break; }
        }
        return fill_concat(converted, radix, max_len);

        pure fn fill_concat(v: &[BigDigit], radix: uint, l: uint) -> ~str {
            if v.is_empty() { return ~"0" }
            str::trim_left_chars(str::concat(vec::reversed(v).map(|n| {
                let s = uint::to_str(*n as uint, radix);
                str::from_chars(vec::from_elem(l - s.len(), '0')) + s
            })), ['0'])
        }
    }

    #[cfg(target_arch = "x86_64")]
    priv pure fn get_radix_base(radix: uint) -> (uint, uint) {
        assert 1 < radix && radix <= 16;
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
            _  => fail
        }
    }

    #[cfg(target_arch = "arm")]
    #[cfg(target_arch = "x86")]
    priv pure fn get_radix_base(radix: uint) -> (uint, uint) {
        assert 1 < radix && radix <= 16;
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
            _  => fail
        }
    }
}

#[cfg(test)]
mod biguint_tests {

    use core::*;
    use core::num::{Num, Zero, One};
    use super::{BigInt, BigUint, BigDigit};

    #[test]
    fn test_from_slice() {
        fn check(slice: &[BigDigit], data: &[BigDigit]) {
            assert data == BigUint::from_slice(slice).data;
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
            for vec::view(data, i, data.len()).eachi |j0, nj| {
                let j = j0 + i;
                if i == j {
                    assert ni.cmp(nj) == 0;
                    assert nj.cmp(ni) == 0;
                    assert ni == nj;
                    assert !(ni != nj);
                    assert ni <= nj;
                    assert ni >= nj;
                    assert !(ni < nj);
                    assert !(ni > nj);
                } else {
                    assert ni.cmp(nj) < 0;
                    assert nj.cmp(ni) > 0;

                    assert !(ni == nj);
                    assert ni != nj;

                    assert ni <= nj;
                    assert !(ni >= nj);
                    assert ni < nj;
                    assert !(ni > nj);

                    assert !(nj <= ni);
                    assert nj >= ni;
                    assert !(nj < ni);
                    assert nj > ni;
                }
            }
        }
    }

    #[test]
    fn test_shl() {
        fn check(v: ~[BigDigit], shift: uint, ans: ~[BigDigit]) {
            assert BigUint::new(v) << shift == BigUint::new(ans);
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
    fn test_shr() {
        fn check(v: ~[BigDigit], shift: uint, ans: ~[BigDigit]) {
            assert BigUint::new(v) >> shift == BigUint::new(ans);
        }

        check(~[], 3, ~[]);
        check(~[1, 1, 1], 3,
              ~[1 << (BigDigit::bits - 3), 1 << (BigDigit::bits - 3)]);
        check(~[1 << 2], 2, ~[1]);
        check(~[1, 2], 3, ~[1 << (BigDigit::bits - 2)]);
        check(~[1, 1, 2], 3 + BigDigit::bits, ~[1 << (BigDigit::bits - 2)]);
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
            assert b == Num::from_int(i);
            assert b.to_int() == i;
        }

        check(~[], 0);
        check(~[1], 1);
        check(~[-1], (uint::max_value >> BigDigit::bits) as int);
        check(~[ 0,  1], ((uint::max_value >> BigDigit::bits) + 1) as int);
        check(~[-1, -1 >> 1], int::max_value);

        assert BigUint::new(~[0, -1]).to_int() == int::max_value;
        assert BigUint::new(~[0, 0, 1]).to_int() == int::max_value;
        assert BigUint::new(~[0, 0, -1]).to_int() == int::max_value;
    }

    #[test]
    fn test_convert_uint() {
        fn check(v: ~[BigDigit], u: uint) {
            let b = BigUint::new(v);
            assert b == BigUint::from_uint(u);
            assert b.to_uint() == u;
        }

        check(~[], 0);
        check(~[ 1], 1);
        check(~[-1], uint::max_value >> BigDigit::bits);
        check(~[ 0,  1], (uint::max_value >> BigDigit::bits) + 1);
        check(~[ 0, -1], uint::max_value << BigDigit::bits);
        check(~[-1, -1], uint::max_value);

        assert BigUint::new(~[0, 0, 1]).to_uint()  == uint::max_value;
        assert BigUint::new(~[0, 0, -1]).to_uint() == uint::max_value;
    }

    const sum_triples: &[(&[BigDigit], &[BigDigit], &[BigDigit])] = &[
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

            assert a + b == c;
            assert b + a == c;
        }
    }

    #[test]
    fn test_sub() {
        for sum_triples.each |elm| {
            let (aVec, bVec, cVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);

            assert c - a == b;
            assert c - b == a;
        }
    }

    const mul_triples: &[(&[BigDigit], &[BigDigit], &[BigDigit])] = &[
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

    const divmod_quadruples: &[(&[BigDigit], &[BigDigit],
                                &[BigDigit], &[BigDigit])]
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

            assert a * b == c;
            assert b * a == c;
        }

        for divmod_quadruples.each |elm| {
            let (aVec, bVec, cVec, dVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);
            let d = BigUint::from_slice(dVec);

            assert a == b * c + d;
            assert a == c * b + d;
        }
    }

    #[test]
    fn test_divmod() {
        for mul_triples.each |elm| {
            let (aVec, bVec, cVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);

            if a.is_not_zero() {
                assert c.divmod(&a) == (b, Zero::zero());
            }
            if b.is_not_zero() {
                assert c.divmod(&b) == (a, Zero::zero());
            }
        }

        for divmod_quadruples.each |elm| {
            let (aVec, bVec, cVec, dVec) = *elm;
            let a = BigUint::from_slice(aVec);
            let b = BigUint::from_slice(bVec);
            let c = BigUint::from_slice(cVec);
            let d = BigUint::from_slice(dVec);

            if b.is_not_zero() { assert a.divmod(&b) == (c, d); }
        }
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
                32 => ~"8589934593", 16 => ~"131073", _ => fail
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
                _ => fail
            }),
            (16, ~"3" +
             str::from_chars(vec::from_elem(bits / 4 - 1, '0')) + "2" +
             str::from_chars(vec::from_elem(bits / 4 - 1, '0')) + "1")
        ]), ( BigUint::from_slice(~[ BigDigit::max_value ]), ~[
            (2, str::from_chars(vec::from_elem(bits, '1'))),
            (4, str::from_chars(vec::from_elem(bits / 2, '3'))),
            (16, str::from_chars(vec::from_elem(bits / 4, 'f')))
        ]), ( BigUint::from_slice(~[
            BigDigit::max_value, BigDigit::max_value
        ]), ~[
            (2, str::from_chars(vec::from_elem(2 * bits, '1'))),
            (4, str::from_chars(vec::from_elem(2 * bits / 2, '3'))),
            (16, str::from_chars(vec::from_elem(2 * bits / 4, 'f')))
        ]) ]
    }

    #[test]
    fn test_to_str_radix() {
        for to_str_pairs().each |num_pair| {
            let &(n, rs) = num_pair;
            for rs.each |str_pair| {
                let &(radix, str) = str_pair;
                assert n.to_str_radix(radix) == str;
            }
        }
    }

    #[test]
    fn test_from_str_radix() {
        for to_str_pairs().each |num_pair| {
            let &(n, rs) = num_pair;
            for rs.each |str_pair| {
                let &(radix, str) = str_pair;
                assert Some(n) == BigUint::from_str_radix(str, radix);
            }
        }

        assert BigUint::from_str_radix(~"Z", 10) == None;
        assert BigUint::from_str_radix(~"_", 2) == None;
        assert BigUint::from_str_radix(~"-1", 10) == None;
    }

    #[test]
    fn test_factor() {
        fn factor(n: uint) -> BigUint {
            let mut f= One::one::<BigUint>();
            for uint::range(2, n + 1) |i| {
                f *= BigUint::from_uint(i);
            }
            return f;
        }

        fn check(n: uint, s: &str) {
            let n = factor(n);
            let ans = match BigUint::from_str_radix(s, 10) {
                Some(x) => x, None => fail
            };
            assert n == ans;
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
    use core::num::{Num, Zero, One};

    #[test]
    fn test_from_biguint() {
        fn check(inp_s: Sign, inp_n: uint, ans_s: Sign, ans_n: uint) {
            let inp = BigInt::from_biguint(inp_s, BigUint::from_uint(inp_n));
            let ans = BigInt { sign: ans_s, data: BigUint::from_uint(ans_n)};
            assert inp == ans;
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
            for vec::view(nums, i, nums.len()).eachi |j0, nj| {
                let j = i + j0;
                if i == j {
                    assert ni.cmp(nj) == 0;
                    assert nj.cmp(ni) == 0;
                    assert ni == nj;
                    assert !(ni != nj);
                    assert ni <= nj;
                    assert ni >= nj;
                    assert !(ni < nj);
                    assert !(ni > nj);
                } else {
                    assert ni.cmp(nj) < 0;
                    assert nj.cmp(ni) > 0;

                    assert !(ni == nj);
                    assert ni != nj;

                    assert ni <= nj;
                    assert !(ni >= nj);
                    assert ni < nj;
                    assert !(ni > nj);

                    assert !(nj <= ni);
                    assert nj >= ni;
                    assert !(nj < ni);
                    assert nj > ni;
                }
            }
        }
    }

    #[test]
    fn test_convert_int() {
        fn check(b: BigInt, i: int) {
            assert b == Num::from_int(i);
            assert b.to_int() == i;
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);
        check(BigInt::from_biguint(
            Plus, BigUint::from_uint(int::max_value as uint)
        ), int::max_value);

        assert BigInt::from_biguint(
            Plus, BigUint::from_uint(int::max_value as uint + 1)
        ).to_int() == int::max_value;
        assert BigInt::from_biguint(
            Plus, BigUint::new(~[1, 2, 3])
        ).to_int() == int::max_value;

        check(BigInt::from_biguint(
            Minus, BigUint::from_uint(-int::min_value as uint)
        ), int::min_value);
        assert BigInt::from_biguint(
            Minus, BigUint::from_uint(-int::min_value as uint + 1)
        ).to_int() == int::min_value;
        assert BigInt::from_biguint(
            Minus, BigUint::new(~[1, 2, 3])
        ).to_int() == int::min_value;
    }

    #[test]
    fn test_convert_uint() {
        fn check(b: BigInt, u: uint) {
            assert b == BigInt::from_uint(u);
            assert b.to_uint() == u;
        }

        check(Zero::zero(), 0);
        check(One::one(), 1);

        check(
            BigInt::from_biguint(Plus, BigUint::from_uint(uint::max_value)),
            uint::max_value);
        assert BigInt::from_biguint(
            Plus, BigUint::new(~[1, 2, 3])
        ).to_uint() == uint::max_value;

        assert BigInt::from_biguint(
            Minus, BigUint::from_uint(uint::max_value)
        ).to_uint() == 0;
        assert BigInt::from_biguint(
            Minus, BigUint::new(~[1, 2, 3])
        ).to_uint() == 0;
    }

    const sum_triples: &[(&[BigDigit], &[BigDigit], &[BigDigit])] = &[
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

            assert a + b == c;
            assert b + a == c;
            assert c + (-a) == b;
            assert c + (-b) == a;
            assert a + (-c) == (-b);
            assert b + (-c) == (-a);
            assert (-a) + (-b) == (-c);
            assert a + (-a) == Zero::zero();
        }
    }

    #[test]
    fn test_sub() {
        for sum_triples.each |elm| {
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

            assert c - a == b;
            assert c - b == a;
            assert (-b) - a == (-c);
            assert (-a) - b == (-c);
            assert b - (-a) == c;
            assert a - (-b) == c;
            assert (-c) - (-a) == (-b);
            assert a - a == Zero::zero();
        }
    }

    const mul_triples: &[(&[BigDigit], &[BigDigit], &[BigDigit])] = &[
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

    const divmod_quadruples: &[(&[BigDigit], &[BigDigit],
                                &[BigDigit], &[BigDigit])]
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

            assert a * b == c;
            assert b * a == c;

            assert (-a) * b == -c;
            assert (-b) * a == -c;
        }

        for divmod_quadruples.each |elm| {
            let (aVec, bVec, cVec, dVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);
            let d = BigInt::from_slice(Plus, dVec);

            assert a == b * c + d;
            assert a == c * b + d;
        }
    }

    #[test]
    fn test_divmod() {
        fn check_sub(a: &BigInt, b: &BigInt, ans_d: &BigInt, ans_m: &BigInt) {
            let (d, m) = a.divmod(b);
            if m.is_not_zero() {
                assert m.sign == b.sign;
            }
            assert m.abs() <= b.abs();
            assert *a == b * d + m;
            assert d == *ans_d;
            assert m == *ans_m;
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

            if a.is_not_zero() { check(&c, &a, &b, &Zero::zero()); }
            if b.is_not_zero() { check(&c, &b, &a, &Zero::zero()); }
        }

        for divmod_quadruples.each |elm| {
            let (aVec, bVec, cVec, dVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);
            let d = BigInt::from_slice(Plus, dVec);

            if b.is_not_zero() {
                check(&a, &b, &c, &d);
            }
        }
    }


    #[test]
    fn test_quotrem() {
        fn check_sub(a: &BigInt, b: &BigInt, ans_q: &BigInt, ans_r: &BigInt) {
            let (q, r) = a.quotrem(b);
            if r.is_not_zero() {
                assert r.sign == a.sign;
            }
            assert r.abs() <= b.abs();
            assert *a == b * q + r;
            assert q == *ans_q;
            assert r == *ans_r;
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

            if a.is_not_zero() { check(&c, &a, &b, &Zero::zero()); }
            if b.is_not_zero() { check(&c, &b, &a, &Zero::zero()); }
        }

        for divmod_quadruples.each |elm| {
            let (aVec, bVec, cVec, dVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);
            let d = BigInt::from_slice(Plus, dVec);

            if b.is_not_zero() {
                check(&a, &b, &c, &d);
            }
        }
    }

    #[test]
    fn test_to_str_radix() {
        fn check(n: int, ans: &str) {
            assert ans == Num::from_int::<BigInt>(n).to_str_radix(10);
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
            let ans = ans.map(|&n| Num::from_int(n));
            assert BigInt::from_str_radix(s, 10) == ans;
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
        assert -BigInt::new(Plus,  ~[1, 1, 1]) ==
            BigInt::new(Minus, ~[1, 1, 1]);
        assert -BigInt::new(Minus, ~[1, 1, 1]) ==
            BigInt::new(Plus,  ~[1, 1, 1]);
        assert -Zero::zero::<BigInt>() == Zero::zero::<BigInt>();
    }
}

#[cfg(test)]
mod vec_ops_test {
    use vec_ops, BigDigit;
    use core::*;

    #[test]
    fn test_reduce_zeros() {
        fn check(inp: ~[BigDigit], out: &[BigDigit]) {
            let mut inp = inp;
            vec_ops::reduce_zeros(&mut inp);
            assert vec::eq(inp, out);
        }

        check(~[1, 2, 3], [1, 2, 3]);
        check(~[1, 2, 3, 0, 0, 0], [1, 2, 3]);
        check(~[], []);
        check(~[0, 0], []);
    }

    #[test]
    fn test_cmp() {
        fn check(a: &[BigDigit], b: &[BigDigit], offset: uint, c: int) {
            if c != 0 {
                // c and result have same sign
                assert vec_ops::cmp_offset(a, b, offset) * c > 0;
            } else {
                assert vec_ops::cmp_offset(a, b, offset) == 0;
            }
            if offset == 0 {
                if c != 0 {
                    // c and result have different sign
                    assert vec_ops::cmp_offset(b, a, offset) * c < 0;
                } else {
                    assert vec_ops::cmp_offset(b, a, offset) == 0;
                }
            }
        }

        check(~[], ~[], 0, 0);
        check(~[1, 2, 3], ~[], 0, 1);
        check(~[1, 2, 3], ~[1, 1, 1], 0, 1);
        check(~[1, 2, 3], ~[1, 2, 3], 0, 0);

        check(~[], ~[], 3, -1);
        check(~[1, 2, 3], ~[], 3, 0);
        check(~[1, 2, 3, 4], ~[], 3, 1);
        check(~[1, 2, 3, 4], ~[4], 3, 0);
        check(~[1, 2, 3, 4], ~[5], 3, -1);
    }

    #[test]
    fn test_replace_offset() {
        fn check(buf: ~[BigDigit], inp: &[BigDigit], offset: uint,
                 out: &[BigDigit]) {
            let mut buf = buf;
            vec_ops::replace_offset(&mut buf, inp, offset);
            assert vec::eq(buf, out);
        }

        check(~[], &[], 0, &[]);
        check(~[], &[], 3, &[]);

        check(~[], &[1, 2, 3], 0, &[1, 2, 3]);
        check(~[], &[1, 2, 3], 3, &[0, 0, 0, 1, 2, 3]);

        check(~[1, 2, 3], &[], 0, &[1, 2, 3]);
        check(~[1, 2, 3], &[], 3, &[1, 2, 3]);
        check(~[1, 2, 3], &[4, 5, 6], 0, &[4, 5, 6]);
        check(~[1, 2, 3], &[4, 5, 6], 3, &[1, 2, 3, 4, 5, 6]);
        check(~[1, 2, 3, 0, 0, 0, 7, 8, 9], &[4, 5, 6], 3,
              &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_assign() {
        fn check(dst: ~[BigDigit], src: &[BigDigit]) {
            let mut dst = dst;
            vec_ops::assign(&mut dst, src);
            assert vec::eq(dst, src);
        }
        check(~[], ~[]);
        check(~[], ~[1, 2, 3]);
        check(~[1, 2, 3], ~[]);
        check(~[1, 2, 3], ~[4, 5, 6]);
    }

    #[test]
    fn test_add_sub_offset_assign() {
        fn check(a: &[BigDigit], b: &[BigDigit], offset: uint,
                 sum: &[BigDigit]) {
            let mut v1 = vec::from_slice(a);
            vec_ops::add_offset_assign(&mut v1, b, offset);
            assert vec::eq(v1, sum);

            let mut v1 = vec::from_slice(sum);
            vec_ops::sub_offset_assign(&mut v1, b, offset);
            assert vec::eq(v1, a);

            if offset > 0 {
                let b2 = if b.is_empty() {
                    ~[]
                } else {
                    vec::from_elem(offset, 0) + b
                };

                let mut v2 = vec::from_slice(a);
                vec_ops::add_offset_assign(&mut v2, b2, 0);
                assert vec::eq(v2, sum);

                let mut v2 = vec::from_slice(sum);
                vec_ops::sub_offset_assign(&mut v2, a, 0);
                assert vec::eq(v2, b2);

                let mut v3 = vec::from_slice(b2);
                vec_ops::add_offset_assign(&mut v3, a, 0);
                assert vec::eq(v3, sum);

                let mut v3 = vec::from_slice(sum);
                vec_ops::sub_offset_assign(&mut v3, b2, 0);
                assert vec::eq(v3, a);
            }
        }

        check([], [], 0, []);
        check([1, 2, 3], [], 0, [1, 2, 3]);
        check([], [1, 2, 3], 0, [1, 2, 3]);
        check([BigDigit::max_value], [1], 0, [0, 1]);

        check([], [1, 2, 3], 3, [0, 0, 0, 1, 2, 3]);
        check([1, 2], [2, 3, 4], 1, [1, 4, 3, 4]);
        check([1, 2], [], 3, [1, 2]);
    }

    #[test]
    fn test_add_sub_set() {
        fn check(a: &[BigDigit], b: &[BigDigit], sum: &[BigDigit]) {
            let mut v1 = ~[];
            vec_ops::add_set(&mut v1, a, b);
            assert vec::eq(v1, sum);

            let mut v1 = ~[];
            vec_ops::sub_set(&mut v1, sum, a);
            assert vec::eq(v1, b);

            let mut v2 = ~[];
            vec_ops::add_set(&mut v2, b, a);
            assert vec::eq(v2, sum);

            let mut v2 = ~[];
            vec_ops::sub_set(&mut v2, sum, b);
            assert vec::eq(v2, a);
        }

        check([], [], []);
        check([1, 2, 3], [], [1, 2, 3]);
        check([], [1, 2, 3], [1, 2, 3]);
        check([BigDigit::max_value], [1], [0, 1]);
        check([BigDigit::max_value], [1, BigDigit::max_value], [0, 0, 1]);
    }

    #[test]
    fn test_mul_digit() {
        fn check(a: ~[BigDigit], n: BigDigit, prod: &[BigDigit]) {
            let mut a = a;
            vec_ops::mul_digit_assign(&mut a, n);
            assert vec::eq(a, prod);
        }

        check(~[], 0, &[]);
        check(~[], 1, &[]);
        check(~[1, 2], 0, &[]);
        check(~[1, 2], 1, &[1, 2]);
        check(~[1, 2], 100, &[100, 200]);
        check(~[1, 2], BigDigit::max_value,
              &[BigDigit::max_value, BigDigit::max_value - 1, 1]);
    }


    #[test]
    fn test_mul_digit_set() {
        fn check(a: &[BigDigit], n: BigDigit, prod: &[BigDigit]) {
            let mut v = ~[];
            vec_ops::mul_digit_set(&mut v, a, n);
            assert vec::eq(v, prod);
        }

        check(&[], 0, &[]);
        check(&[], 1, &[]);
        check(&[1, 2], 0, &[]);
        check(&[1, 2], 1, &[1, 2]);
        check(&[1, 2], 100, &[100, 200]);
        check(&[1, 2], BigDigit::max_value,
              &[BigDigit::max_value, BigDigit::max_value - 1, 1]);
    }
}
