/*!

A Big integer (signed version: BigInt, unsigned version: BigUint).

A BigUint is represented as an array of BigDigits.
A BigInt is a combination of BigUint and Sign.
*/

use core::cmp::{Eq, Ord};

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
    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "arm")]
    pub const bits: uint = 16;

    #[cfg(target_arch = "x86_64")]
    pub const bits: uint = 32;

    pub const base: uint = 1 << bits;
    priv const hi_mask: uint = (-1 as uint) << bits;
    priv const lo_mask: uint = (-1 as uint) >> bits;

    priv pure fn get_hi(n: uint) -> BigDigit { (n >> bits) as BigDigit }
    priv pure fn get_lo(n: uint) -> BigDigit { (n & lo_mask) as BigDigit }

    /// Split one machine sized unsigned integer into two BigDigits.
    pub pure fn from_uint(n: uint) -> (BigDigit, BigDigit) {
        (get_hi(n), get_lo(n))
    }

    /// Join two BigDigits into one machine sized unsigned integer
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
    priv data: @[BigDigit]
}

impl BigUint : Eq {
    pure fn eq(&self, other: &BigUint) -> bool { self.cmp(other) == 0 }
    pure fn ne(&self, other: &BigUint) -> bool { self.cmp(other) != 0 }
}

impl BigUint : Ord {
    pure fn lt(&self, other: &BigUint) -> bool { self.cmp(other) <  0 }
    pure fn le(&self, other: &BigUint) -> bool { self.cmp(other) <= 0 }
    pure fn ge(&self, other: &BigUint) -> bool { self.cmp(other) >= 0 }
    pure fn gt(&self, other: &BigUint) -> bool { self.cmp(other) >  0 }
}

impl BigUint : ToStr {
    pure fn to_str() -> ~str { self.to_str_radix(10) }
}

impl BigUint : from_str::FromStr {
    static pure fn from_str(s: &str) -> Option<BigUint> {
        BigUint::from_str_radix(s, 10)
    }
}

impl BigUint : Shl<uint, BigUint> {
    pure fn shl(&self, rhs: &uint) -> BigUint {
        let n_unit = *rhs / BigDigit::bits;
        let n_bits = *rhs % BigDigit::bits;
        return self.shl_unit(n_unit).shl_bits(n_bits);
    }
}

impl BigUint : Shr<uint, BigUint> {
    pure fn shr(&self, rhs: &uint) -> BigUint {
        let n_unit = *rhs / BigDigit::bits;
        let n_bits = *rhs % BigDigit::bits;
        return self.shr_unit(n_unit).shr_bits(n_bits);
    }
}

impl BigUint : Num {
    pure fn add(&self, other: &BigUint) -> BigUint {
        let new_len = uint::max(self.data.len(), other.data.len());

        let mut carry = 0;
        let sum = do at_vec::from_fn(new_len) |i| {
            let ai = if i < self.data.len()  { self.data[i]  } else { 0 };
            let bi = if i < other.data.len() { other.data[i] } else { 0 };
            let (hi, lo) = BigDigit::from_uint(
                (ai as uint) + (bi as uint) + (carry as uint)
            );
            carry = hi;
            lo
        };
        if carry == 0 { return BigUint::from_at_vec(sum) };
        return BigUint::from_at_vec(sum + [carry]);
    }

    pure fn sub(&self, other: &BigUint) -> BigUint {
        let new_len = uint::max(self.data.len(), other.data.len());

        let mut borrow = 0;
        let diff = do at_vec::from_fn(new_len) |i| {
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

        assert borrow == 0;     // <=> assert (self >= other);
        return BigUint::from_at_vec(diff);
    }

    pure fn mul(&self, other: &BigUint) -> BigUint {
        if self.is_zero() || other.is_zero() { return BigUint::zero(); }

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
        let mm =  match (sub_sign(sHi, sLo), sub_sign(oHi, oLo)) {
            ((s1, n1), (s2, n2)) if s1 * s2 < 0 => hh + ll + (n1 * n2),
            ((s1, n1), (s2, n2)) if s1 * s2 > 0 => hh + ll - (n1 * n2),
            _   => hh + ll,
        };

        return ll + mm.shl_unit(half_len) + hh.shl_unit(half_len * 2);

        pure fn mul_digit(a: &BigUint, n: BigDigit) -> BigUint {
            if n == 0 { return BigUint::zero(); }
            if n == 1 { return *a; }

            let mut carry = 0;
            let prod = do at_vec::map(a.data) |ai| {
                let (hi, lo) = BigDigit::from_uint(
                    (*ai as uint) * (n as uint) + (carry as uint)
                );
                carry = hi;
                lo
            };
            if carry == 0 { return BigUint::from_at_vec(prod) };
            return BigUint::from_at_vec(prod + [carry]);
        }

        pure fn cut_at(a: &BigUint, n: uint) -> (BigUint, BigUint) {
            let mid = uint::min(a.data.len(), n);
            return (BigUint::from_slice(vec::view(a.data, mid, a.data.len())),
                    BigUint::from_slice(vec::view(a.data, 0, mid)));
        }

        pure fn sub_sign(a: BigUint, b: BigUint) -> (int, BigUint) {
            match a.cmp(&b) {
                s if s < 0 => (s, b - a),
                s if s > 0 => (s, a - b),
                _          => (0, BigUint::zero())
            }
        }
    }

    pure fn div(&self, other: &BigUint) -> BigUint{
        self.divmod(other).first()
    }
    pure fn modulo(&self, other: &BigUint) -> BigUint {
        self.divmod(other).second()
    }

    pure fn neg(&self) -> BigUint { fail }

    pure fn to_int(&self) -> int {
        uint::min(self.to_uint(), int::max_value as uint) as int
    }

    static pure fn from_int(n: int) -> BigUint {
        if (n < 0) { BigUint::zero() } else { BigUint::from_uint(n as uint) }
    }
}

pub impl BigUint {
    /// Creates and initializes an BigUint.
    static pub pure fn from_uint(n: uint) -> BigUint {
        match BigDigit::from_uint(n) {
            (0,  0)  => BigUint::zero(),
            (0,  n0) => BigUint::from_at_vec(@[n0]),
            (n1, n0) => BigUint::from_at_vec(@[n0, n1])
        }
    }

    /// Creates and initializes an BigUint.
    static pub pure fn from_slice(slice: &[BigDigit]) -> BigUint {
        // omit trailing zeros
        let new_len = slice.rposition(|n| *n != 0)
            .map_default(0, |p| *p + 1);

        return BigUint { data: at_vec::from_fn(new_len, |i| slice[i]) };
    }

    /// Creates and initializes an BigUint.
    static pub pure fn from_at_vec(at_vec: @[BigDigit]) -> BigUint {
        // omit trailing zeros
        let new_len = at_vec.rposition(|n| *n != 0)
            .map_default(0, |p| *p + 1);

        if new_len == at_vec.len() { return BigUint { data: at_vec }; }
        return BigUint { data: at_vec::from_fn(new_len, |i| at_vec[i]) };
    }

    /// Creates and initializes an BigUint.
    static pub pure fn from_str_radix(s: &str, radix: uint)
        -> Option<BigUint> {
        BigUint::parse_bytes(str::to_bytes(s), radix)
    }

    /// Creates and initializes an BigUint.
    static pub pure fn parse_bytes(buf: &[u8], radix: uint)
        -> Option<BigUint> {
        let (base, unit_len) = get_radix_base(radix);
        let base_num: BigUint = BigUint::from_uint(base);

        let mut end             = buf.len();
        let mut n: BigUint      = BigUint::zero();
        let mut power: BigUint  = BigUint::one();
        loop {
            let start = uint::max(end, unit_len) - unit_len;
            match uint::parse_bytes(vec::view(buf, start, end), radix) {
                Some(d) => n += BigUint::from_uint(d) * power,
                None    => return None
            }
            if end <= unit_len {
                return Some(n);
            }
            end -= unit_len;
            power *= base_num;
        }
    }

    static pub pure fn zero() -> BigUint { BigUint::from_at_vec(@[]) }
    static pub pure fn one() -> BigUint { BigUint::from_at_vec(@[1]) }
    pure fn abs(&self) -> BigUint { *self }

    /// Compare two BigUint value.
    pure fn cmp(&self, other: &BigUint) -> int {
        let s_len = self.data.len(), o_len = other.data.len();
        if s_len < o_len { return -1; }
        if s_len > o_len { return  1;  }

        for vec::rev_eachi(self.data) |i, elm| {
            match (*elm, other.data[i]) {
                (l, r) if l < r => return -1,
                (l, r) if l > r => return  1,
                _               => loop
            };
        }
        return 0;
    }

    pure fn divmod(&self, other: &BigUint) -> (BigUint, BigUint) {
        if other.is_zero() { fail }
        if self.is_zero() { return (BigUint::zero(), BigUint::zero()); }
        if *other == BigUint::one() { return (*self, BigUint::zero()); }

        match self.cmp(other) {
            s if s < 0 => return (BigUint::zero(), *self),
            0          => return (BigUint::one(), BigUint::zero()),
            _          => {} // Do nothing
        }

        let mut shift = 0;
        let mut n = other.data.last();
        while n < (1 << BigDigit::bits - 2) {
            n <<= 1;
            shift += 1;
        }
        assert shift < BigDigit::bits;
        let (d, m) = divmod_inner(self << shift, other << shift);
        return (d, m >> shift);


        pure fn divmod_inner(a: BigUint, b: BigUint) -> (BigUint, BigUint) {
            let mut r = a;
            let mut d = BigUint::zero();
            let mut n = 1;
            while r >= b {
                let mut (d0, d_unit, b_unit) = div_estimate(r, b, n);
                let mut prod = b * d0;
                while prod > r {
                    d0   -= d_unit;
                    prod -= b_unit;
                }
                if d0.is_zero() {
                    n = 2;
                    loop;
                }
                n = 1;
                d += d0;
                r -= prod;
            }
            return (d, r);
        }

        pure fn div_estimate(a: BigUint, b: BigUint, n: uint)
            -> (BigUint, BigUint, BigUint) {
            if a.data.len() < n {
                return (BigUint::zero(), BigUint::zero(), a);
            }

            let an = vec::view(a.data, a.data.len() - n, a.data.len());
            let bn = b.data.last();
            let mut d = ~[];
            let mut carry = 0;
            for vec::rev_each(an) |elt| {
                let ai = BigDigit::to_uint(carry, *elt);
                let di = ai / (bn as uint);
                assert di < BigDigit::base;
                carry = (ai % (bn as uint)) as BigDigit;
                d = ~[di as BigDigit] + d;
            }

            let shift = (a.data.len() - an.len()) - (b.data.len() - 1);
            return (BigUint::from_slice(d).shl_unit(shift),
                    BigUint::one().shl_unit(shift),
                    b.shl_unit(shift));
        }
    }

    pure fn quot(&self, other: &BigUint) -> BigUint {
        self.quotrem(other).first()
    }
    pure fn rem(&self, other: &BigUint) -> BigUint {
        self.quotrem(other).second()
    }
    pure fn quotrem(&self, other: &BigUint) -> (BigUint, BigUint) {
        self.divmod(other)
    }

    pure fn is_zero(&self) -> bool { self.data.is_empty() }
    pure fn is_not_zero(&self) -> bool { self.data.is_not_empty() }
    pure fn is_positive(&self) -> bool { self.is_not_zero() }
    pure fn is_negative(&self) -> bool { false }
    pure fn is_nonpositive(&self) -> bool { self.is_zero() }
    pure fn is_nonnegative(&self) -> bool { true }

    pure fn to_uint(&self) -> uint {
        match self.data.len() {
            0 => 0,
            1 => self.data[0] as uint,
            2 => BigDigit::to_uint(self.data[1], self.data[0]),
            _ => uint::max_value
        }
    }

    pure fn to_str_radix(&self, radix: uint) -> ~str {
        assert 1 < radix && radix <= 16;

        pure fn convert_base(n: &BigUint, base: uint) -> @[BigDigit] {
            if base == BigDigit::base { return n.data; }
            let divider    = BigUint::from_uint(base);
            let mut result = @[];
            let mut r      = *n;
            while r > divider {
                let (d, r0) = r.divmod(&divider);
                result += [r0.to_uint() as BigDigit];
                r = d;
            }
            if r.is_not_zero() {
                result += [r.to_uint() as BigDigit];
            }
            return result;
        }

        pure fn fill_concat(v: &[BigDigit], radix: uint, l: uint) -> ~str {
            if v.is_empty() { return ~"0" }
            str::trim_left_chars(str::concat(vec::reversed(v).map(|n| {
                let s = uint::to_str(*n as uint, radix);
                str::from_chars(vec::from_elem(l - s.len(), '0')) + s
            })), ['0'])
        }

        let (base, max_len) = get_radix_base(radix);
        return fill_concat(convert_base(self, base), radix, max_len);
    }

    priv pure fn shl_unit(&self, n_unit: uint) -> BigUint {
        if n_unit == 0 || self.is_zero() { return *self; }

        return BigUint::from_at_vec(at_vec::from_elem(n_unit, 0) + self.data);
    }

    priv pure fn shl_bits(&self, n_bits: uint) -> BigUint {
        if n_bits == 0 || self.is_zero() { return *self; }

        let mut carry = 0;
        let shifted = do at_vec::map(self.data) |elem| {
            let (hi, lo) = BigDigit::from_uint(
                (*elem as uint) << n_bits | (carry as uint)
            );
            carry = hi;
            lo
        };
        if carry == 0 { return BigUint::from_at_vec(shifted); }
        return BigUint::from_at_vec(shifted + [carry]);
    }

    priv pure fn shr_unit(&self, n_unit: uint) -> BigUint {
        if n_unit == 0 { return *self; }
        if self.data.len() < n_unit { return BigUint::zero(); }
        return BigUint::from_slice(
            vec::view(self.data, n_unit, self.data.len())
        );
    }

    priv pure fn shr_bits(&self, n_bits: uint) -> BigUint {
        if n_bits == 0 || self.data.is_empty() { return *self; }

        let mut borrow = 0;
        let mut shifted = @[];
        for vec::rev_each(self.data) |elem| {
            // internal compiler error: no enclosing scope with id 10671
            // shifted = @[(*elem >> n_bits) | borrow] + shifted;
            shifted = at_vec::append(@[(*elem >> n_bits) | borrow], shifted);
            borrow = *elem << (uint::bits - n_bits);
        }
        return BigUint::from_at_vec(shifted);
    }
}

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

/// A Sign is a BigInt's composing element.
pub enum Sign { Minus, Zero, Plus }

impl Sign : Eq {
    pure fn eq(&self, other: &Sign) -> bool { self.cmp(other) == 0 }
    pure fn ne(&self, other: &Sign) -> bool { self.cmp(other) != 0 }
}

impl Sign : Ord {
    pure fn lt(&self, other: &Sign) -> bool { self.cmp(other) <  0 }
    pure fn le(&self, other: &Sign) -> bool { self.cmp(other) <= 0 }
    pure fn ge(&self, other: &Sign) -> bool { self.cmp(other) >= 0 }
    pure fn gt(&self, other: &Sign) -> bool { self.cmp(other) >  0 }
}

pub impl Sign {
    /// Compare two Sign.
    pure fn cmp(&self, other: &Sign) -> int {
        match (*self, *other) {
          (Minus, Minus) | (Zero,  Zero) | (Plus, Plus) =>  0,
          (Minus, Zero)  | (Minus, Plus) | (Zero, Plus) => -1,
          _                                             =>  1
        }
    }

    /// Negate Sign value.
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
    pure fn eq(&self, other: &BigInt) -> bool { self.cmp(other) == 0 }
    pure fn ne(&self, other: &BigInt) -> bool { self.cmp(other) != 0 }
}

impl BigInt : Ord {
    pure fn lt(&self, other: &BigInt) -> bool { self.cmp(other) <  0 }
    pure fn le(&self, other: &BigInt) -> bool { self.cmp(other) <= 0 }
    pure fn ge(&self, other: &BigInt) -> bool { self.cmp(other) >= 0 }
    pure fn gt(&self, other: &BigInt) -> bool { self.cmp(other) >  0 }
}

impl BigInt : ToStr {
    pure fn to_str() -> ~str { self.to_str_radix(10) }
}

impl BigInt : from_str::FromStr {
    static pure fn from_str(s: &str) -> Option<BigInt> {
        BigInt::from_str_radix(s, 10)
    }
}

impl BigInt : Shl<uint, BigInt> {
    pure fn shl(&self, rhs: &uint) -> BigInt {
        BigInt::from_biguint(self.sign, self.data << *rhs)
    }
}

impl BigInt : Shr<uint, BigInt> {
    pure fn shr(&self, rhs: &uint) -> BigInt {
        BigInt::from_biguint(self.sign, self.data >> *rhs)
    }
}

impl BigInt : Num {
    pure fn add(&self, other: &BigInt) -> BigInt {
        match (self.sign, other.sign) {
            (Zero, _)      => *other,
            (_,    Zero)   => *self,
            (Plus, Plus)   => BigInt::from_biguint(Plus,
                                                   self.data + other.data),
            (Plus, Minus)  => self - (-*other),
            (Minus, Plus)  => other - (-*self),
            (Minus, Minus) => -((-self) + (-*other))
        }
    }
    pure fn sub(&self, other: &BigInt) -> BigInt {
        match (self.sign, other.sign) {
            (Zero, _)    => -other,
            (_,    Zero) => *self,
            (Plus, Plus) => match self.data.cmp(&other.data) {
                s if s < 0 =>
                    BigInt::from_biguint(Minus, other.data - self.data),
                s if s > 0 =>
                    BigInt::from_biguint(Plus, self.data - other.data),
                _ =>
                    BigInt::zero()
            },
            (Plus, Minus) => self + (-*other),
            (Minus, Plus) => -((-self) + *other),
            (Minus, Minus) => (-other) - (-*self)
        }
    }
    pure fn mul(&self, other: &BigInt) -> BigInt {
        match (self.sign, other.sign) {
            (Zero, _)     | (_,     Zero)  => BigInt::zero(),
            (Plus, Plus)  | (Minus, Minus) => {
                BigInt::from_biguint(Plus, self.data * other.data)
            },
            (Plus, Minus) | (Minus, Plus) => {
                BigInt::from_biguint(Minus, self.data * other.data)
            }
        }
    }
    pure fn div(&self, other: &BigInt) -> BigInt {
        self.divmod(other).first()
    }
    pure fn modulo(&self, other: &BigInt) -> BigInt {
        self.divmod(other).second()
    }
    pure fn neg(&self) -> BigInt {
        BigInt::from_biguint(self.sign.neg(), self.data)
    }

    pure fn to_int(&self) -> int {
        match self.sign {
            Plus  => uint::min(self.to_uint(), int::max_value as uint) as int,
            Zero  => 0,
            Minus => uint::min((-self).to_uint(),
                               (int::max_value as uint) + 1) as int
        }
    }

    static pure fn from_int(n: int) -> BigInt {
        if n > 0 {
           return BigInt::from_biguint(Plus,  BigUint::from_uint(n as uint));
        }
        if n < 0 {
            return BigInt::from_biguint(
                Minus, BigUint::from_uint(uint::max_value - (n as uint) + 1)
            );
        }
        return BigInt::zero();
    }
}

pub impl BigInt {
    /// Creates and initializes an BigInt.
    static pub pure fn from_biguint(sign: Sign, data: BigUint) -> BigInt {
        if sign == Zero || data.is_zero() {
            return BigInt { sign: Zero, data: BigUint::zero() };
        }
        return BigInt { sign: sign, data: data };
    }

    /// Creates and initializes an BigInt.
    static pub pure fn from_uint(n: uint) -> BigInt {
        if n == 0 { return BigInt::zero(); }
        return BigInt::from_biguint(Plus, BigUint::from_uint(n));
    }

    /// Creates and initializes an BigInt.
    static pub pure fn from_slice(sign: Sign, slice: &[BigDigit]) -> BigInt {
        BigInt::from_biguint(sign, BigUint::from_slice(slice))
    }

    /// Creates and initializes an BigInt.
    static pub pure fn from_at_vec(sign: Sign, at_vec: @[BigDigit])
        -> BigInt {
        BigInt::from_biguint(sign, BigUint::from_at_vec(at_vec))
    }

    /// Creates and initializes an BigInt.
    static pub pure fn from_str_radix(s: &str, radix: uint)
        -> Option<BigInt> {
        BigInt::parse_bytes(str::to_bytes(s), radix)
    }

    /// Creates and initializes an BigInt.
    static pub pure fn parse_bytes(buf: &[u8], radix: uint)
        -> Option<BigInt> {
        if buf.is_empty() { return None; }
        let mut sign  = Plus;
        let mut start = 0;
        if buf[0] == ('-' as u8) {
            sign  = Minus;
            start = 1;
        }
        return BigUint::parse_bytes(vec::view(buf, start, buf.len()), radix)
            .map(|bu| BigInt::from_biguint(sign, *bu));
    }

    static pub pure fn zero() -> BigInt {
        BigInt::from_biguint(Zero, BigUint::zero())
    }
    static pub pure fn one() -> BigInt {
        BigInt::from_biguint(Plus, BigUint::one())
    }

    pure fn abs(&self) -> BigInt { BigInt::from_biguint(Plus, self.data) }

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

    pure fn divmod(&self, other: &BigInt) -> (BigInt, BigInt) {
        // m.sign == other.sign
        let (d_ui, m_ui) = self.data.divmod(&other.data);
        let d = BigInt::from_biguint(Plus, d_ui),
            m = BigInt::from_biguint(Plus, m_ui);
        match (self.sign, other.sign) {
            (_,    Zero)   => fail,
            (Plus, Plus)  | (Zero, Plus)  => (d, m),
            (Plus, Minus) | (Zero, Minus) => if m.is_zero() {
                (-d, BigInt::zero())
            } else {
                (-d - BigInt::one(), m + *other)
            },
            (Minus, Plus) => if m.is_zero() {
                (-d, BigInt::zero())
            } else {
                (-d - BigInt::one(), other - m)
            },
            (Minus, Minus) => (d, -m)
        }
    }

    pure fn quot(&self, other: &BigInt) -> BigInt {
        self.quotrem(other).first()
    }
    pure fn rem(&self, other: &BigInt) -> BigInt {
        self.quotrem(other).second()
    }

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

    pure fn is_zero(&self) -> bool { self.sign == Zero }
    pure fn is_not_zero(&self) -> bool { self.sign != Zero }
    pure fn is_positive(&self) -> bool { self.sign == Plus }
    pure fn is_negative(&self) -> bool { self.sign == Minus }
    pure fn is_nonpositive(&self) -> bool { self.sign != Plus }
    pure fn is_nonnegative(&self) -> bool { self.sign != Minus }

    pure fn to_uint(&self) -> uint {
        match self.sign {
            Plus  => self.data.to_uint(),
            Zero  => 0,
            Minus => 0
        }
    }

    pure fn to_str_radix(&self, radix: uint) -> ~str {
        match self.sign {
            Plus  => self.data.to_str_radix(radix),
            Zero  => ~"0",
            Minus => ~"-" + self.data.to_str_radix(radix)
        }
    }
}

#[cfg(test)]
mod biguint_tests {
    #[test]
    fn test_from_slice() {
        let pairs = [
            (&[1],                &[1]),
            (&[0, 0],             &[]),
            (&[1, 2, 0, 0],       &[1, 2]),
            (&[0, 0, 1, 2, 0, 0], &[0, 0, 1, 2]),
            (&[-1],               &[-1])
        ];
        for pairs.each |p| {
            assert p.second() == BigUint::from_slice(p.first()).data;
        }
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
        assert BigUint::from_at_vec(@[]) << 3 == BigUint::from_at_vec(@[]);
        assert BigUint::from_at_vec(@[1, 1, 1]) << 3 ==
            BigUint::from_at_vec(@[1 << 3, 1 << 3, 1 << 3]);

        assert BigUint::from_at_vec(@[1 << (BigDigit::bits - 2)]) << 2 ==
            BigUint::from_at_vec(@[0, 1]);
        assert BigUint::from_at_vec(@[1 << (BigDigit::bits - 2)]) << 3 ==
            BigUint::from_at_vec(@[0, 2]);
        assert (BigUint::from_at_vec(@[1 << (BigDigit::bits - 2)])
                << (3 + BigDigit::bits)) ==
            BigUint::from_at_vec(@[0, 0, 2]);

        assert BigUint::from_at_vec(
            @[0x7654_3210, 0xfedc_ba98, 0x7654_3210, 0xfedc_ba98]
        ) << 4 == BigUint::from_at_vec(
            @[0x6543_2100, 0xedcb_a987, 0x6543_210f, 0xedcb_a987, 0xf]
        );
        assert BigUint::from_at_vec(
            @[0x2222_1111, 0x4444_3333, 0x6666_5555, 0x8888_7777]
        ) << 16 == BigUint::from_at_vec(
                @[0x1111_0000, 0x3333_2222, 0x5555_4444, 0x7777_6666, 0x8888]
            );
    }

    #[test]
    fn test_shr() {
        assert BigUint::from_at_vec(@[]) >> 3 == BigUint::from_at_vec(@[]);
        assert BigUint::from_at_vec(@[1, 1, 1]) >> 3 == BigUint::from_at_vec(
            @[1 << (BigDigit::bits - 3), 1 << (BigDigit::bits - 3)]
        );

        assert BigUint::from_at_vec(@[1 << 2]) >> 2 ==
            BigUint::from_at_vec(@[1]);
        assert BigUint::from_at_vec(@[1, 2]) >> 3 ==
            BigUint::from_at_vec(@[1 << (BigDigit::bits - 2)]);
        assert BigUint::from_at_vec(@[1, 1, 2]) >> (3 + BigDigit::bits) ==
            BigUint::from_at_vec(@[1 << (BigDigit::bits - 2)]);

        assert BigUint::from_at_vec(
            @[0x6543_2100, 0xedcb_a987, 0x6543_210f, 0xedcb_a987, 0xf]
        ) >> 4 == BigUint::from_at_vec(
            @[0x7654_3210, 0xfedc_ba98, 0x7654_3210, 0xfedc_ba98]
        );

        assert BigUint::from_at_vec(
            @[0x1111_0000, 0x3333_2222, 0x5555_4444, 0x7777_6666, 0x8888]
        ) >> 16 == BigUint::from_at_vec(
            @[0x2222_1111, 0x4444_3333, 0x6666_5555, 0x8888_7777]
        );
    }

    #[test]
    fn test_convert_int() {
        fn check_conv(b: BigUint, i: int) {
            assert b == num::Num::from_int(i);
            assert b.to_int() == i;
        }

        check_conv(BigUint::zero(), 0);
        check_conv(BigUint::from_at_vec(@[1]), 1);

        check_conv(BigUint::from_at_vec(@[-1]),
                   (uint::max_value >> BigDigit::bits) as int);
        check_conv(BigUint::from_at_vec(@[ 0,  1]),
                   ((uint::max_value >> BigDigit::bits) + 1) as int);
        check_conv(BigUint::from_at_vec(@[-1, -1 >> 1]),
                   int::max_value);

        assert BigUint::from_at_vec(@[0, -1]).to_int() == int::max_value;
        assert BigUint::from_at_vec(@[0, 0, 1]).to_int() == int::max_value;
        assert BigUint::from_at_vec(@[0, 0, -1]).to_int() == int::max_value;
    }

    #[test]
    fn test_convert_uint() {
        fn check_conv(b: BigUint, u: uint) {
            assert b == BigUint::from_uint(u);
            assert b.to_uint() == u;
        }

        check_conv(BigUint::zero(), 0);
        check_conv(BigUint::from_at_vec(@[ 1]), 1);
        check_conv(BigUint::from_at_vec(@[-1]),
                   uint::max_value >> BigDigit::bits);
        check_conv(BigUint::from_at_vec(@[ 0,  1]),
                   (uint::max_value >> BigDigit::bits) + 1);
        check_conv(BigUint::from_at_vec(@[ 0, -1]),
                   uint::max_value << BigDigit::bits);
        check_conv(BigUint::from_at_vec(@[-1, -1]),
                   uint::max_value);

        assert BigUint::from_at_vec(@[0, 0, 1]).to_uint()  == uint::max_value;
        assert BigUint::from_at_vec(@[0, 0, -1]).to_uint() == uint::max_value;
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
                assert c.divmod(&a) == (b, BigUint::zero());
            }
            if b.is_not_zero() {
                assert c.divmod(&b) == (a, BigUint::zero());
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
        ~[( BigUint::zero(), ~[
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
            (2,  ~"10" + str::from_chars(vec::from_elem(31, '0')) + "1"),
            (4,  ~"2"  + str::from_chars(vec::from_elem(15, '0')) + "1"),
            (10, ~"8589934593"),
            (16, ~"2"  + str::from_chars(vec::from_elem(7, '0')) + "1")
        ]), (BigUint::from_slice([ 1, 2, 3 ]), ~[
            (2,  ~"11" + str::from_chars(vec::from_elem(30, '0')) + "10" +
             str::from_chars(vec::from_elem(31, '0')) + "1"),
            (4,  ~"3"  + str::from_chars(vec::from_elem(15, '0')) + "2"  +
             str::from_chars(vec::from_elem(15, '0')) + "1"),
            (10, ~"55340232229718589441"),
            (16, ~"3"  + str::from_chars(vec::from_elem(7, '0')) + "2"  +
             str::from_chars(vec::from_elem(7, '0')) + "1")
        ])]
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
            let mut f= BigUint::one();
            for uint::range(2, n + 1) |i| {
                f *= BigUint::from_uint(i);
            }
            return f;
        }

        assert factor(3) == BigUint::from_str_radix(~"6", 10).get();
        assert factor(10) == BigUint::from_str_radix(~"3628800", 10).get();
        assert factor(20) == BigUint::from_str_radix(
            ~"2432902008176640000", 10).get();
        assert factor(30) == BigUint::from_str_radix(
            ~"265252859812191058636308480000000", 10).get();
    }
}

#[cfg(test)]
mod bigint_tests {
    #[test]
    fn test_from_biguint() {
        assert BigInt::from_biguint(Plus, BigUint::from_uint(1)) ==
            BigInt { sign: Plus, data: BigUint::from_uint(1) };
        assert BigInt::from_biguint(Plus, BigUint::zero()) ==
            BigInt { sign: Zero, data: BigUint::zero() };
        assert BigInt::from_biguint(Minus, BigUint::from_uint(1)) ==
            BigInt { sign: Minus, data: BigUint::from_uint(1) };
        assert BigInt::from_biguint(Zero, BigUint::from_uint(1)) ==
            BigInt { sign: Zero, data: BigUint::zero() };
    }

    #[test]
    fn test_cmp() {
        let uints = [ &[2], &[1, 1], &[2, 1], &[1, 1, 1] ]
            .map(|data| BigUint::from_slice(*data));
        let nums: ~[BigInt]
            = vec::reversed(uints).map(|bu| BigInt::from_biguint(Minus, *bu))
            + [ BigInt::zero() ]
            + uints.map(|bu| BigInt::from_biguint(Plus, *bu));

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
        fn check_conv(b: BigInt, i: int) {
            assert b == num::Num::from_int(i);
            assert b.to_int() == i;
        }

        check_conv(BigInt::zero(), 0);
        check_conv(BigInt::one(), 1);
        check_conv(
            BigInt::from_biguint(
                Plus, BigUint::from_uint(int::max_value as uint)),
            int::max_value);

        assert BigInt::from_biguint(
            Plus, BigUint::from_uint(int::max_value as uint + 1)
        ).to_int() == int::max_value;
        assert BigInt::from_biguint(
            Plus, BigUint::from_at_vec(@[1, 2, 3])
        ).to_int() == int::max_value;

        check_conv(
            BigInt::from_biguint(
                Minus, BigUint::from_uint(-int::min_value as uint)),
            int::min_value);
        assert BigInt::from_biguint(
            Minus, BigUint::from_uint(-int::min_value as uint + 1)
        ).to_int() == int::min_value;
        assert BigInt::from_biguint(
            Minus, BigUint::from_at_vec(@[1, 2, 3])
        ).to_int() == int::min_value;
    }

    #[test]
    fn test_convert_uint() {
        fn check_conv(b: BigInt, u: uint) {
            assert b == BigInt::from_uint(u);
            assert b.to_uint() == u;
        }

        check_conv(BigInt::zero(), 0);
        check_conv(BigInt::one(), 1);

        check_conv(
            BigInt::from_biguint(Plus, BigUint::from_uint(uint::max_value)),
            uint::max_value);
        assert BigInt::from_biguint(
            Plus, BigUint::from_at_vec(@[1, 2, 3])
        ).to_uint() == uint::max_value;

        assert BigInt::from_biguint(
            Minus, BigUint::from_uint(uint::max_value)
        ).to_uint() == 0;
        assert BigInt::from_biguint(
            Minus, BigUint::from_at_vec(@[1, 2, 3])
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
            assert a + (-a) == BigInt::zero();
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
            assert a - a == BigInt::zero();
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
        fn check_divmod_sub(a: BigInt, b: BigInt) {
            let (d, m) = a.divmod(&b);
            if m.is_not_zero() {
                assert m.sign == b.sign;
            }
            assert m.abs() <= b.abs();
            assert a == b * d + m;
        }
        fn check_divmod(a: BigInt, b: BigInt, c: BigInt, d: BigInt) {
            check_divmod_sub(a, b);
            check_divmod_sub(a, -b);
            check_divmod_sub(-a, b);
            check_divmod_sub(-a, -b);

            if d.is_zero() {
                assert a.divmod(&b)     == (c, BigInt::zero());
                assert (-a).divmod(&b)  == (-c, BigInt::zero());
                assert (a).divmod(&-b)  == (-c, BigInt::zero());
                assert (-a).divmod(&-b) == (c, BigInt::zero());
            } else {
                // a == bc + d
                assert a.divmod(&b) == (c, d);
                // a == (-b)(-c - 1) + (d - b)
                assert a.divmod(&-b) == (-c - BigInt::one(), d - b);
                // (-a) == b (-c - 1) + (b - d)
                assert (-a).divmod(&b) == (-c - BigInt::one(), b - d);
                // (-a) == (-b)(c) - d
                assert (-a).divmod(&-b) == (c, -d);
            }
        }
        for mul_triples.each |elm| {
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

            if a.is_not_zero() { check_divmod(c, a, b, BigInt::zero()); }
            if b.is_not_zero() { check_divmod(c, b, a, BigInt::zero()); }
        }

        for divmod_quadruples.each |elm| {
            let (aVec, bVec, cVec, dVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);
            let d = BigInt::from_slice(Plus, dVec);

            if b.is_not_zero() {
                check_divmod(a, b, c, d);
            }
        }
    }


    #[test]
    fn test_quotrem() {
        fn check_quotrem_sub(a: BigInt, b: BigInt) {
            let (q, r) = a.quotrem(&b);
            if r.is_not_zero() {
                assert r.sign == a.sign;
            }
            assert r.abs() <= b.abs();
            assert a == b * q + r;
        }
        fn check_quotrem(a: BigInt, b: BigInt, c: BigInt, d: BigInt) {
            check_quotrem_sub(a, b);
            check_quotrem_sub(a, -b);
            check_quotrem_sub(-a, b);
            check_quotrem_sub(-a, -b);

            if d.is_zero() {
                assert a.quotrem(&b)     == (c, BigInt::zero());
                assert (-a).quotrem(&b)  == (-c, BigInt::zero());
                assert (a).quotrem(&-b)  == (-c, BigInt::zero());
                assert (-a).quotrem(&-b) == (c, BigInt::zero());
            } else {
                // a == bc + d
                assert a.quotrem(&b) == (c, d);
                // a == (-b)(-c) + d
                assert a.quotrem(&-b) == (-c, d);
                // (-a) == b (-c) + (-d)
                assert (-a).quotrem(&b) == (-c, -d);
                // (-a) == (-b)(c) - d
                assert (-a).quotrem(&-b) == (c, -d);
            }
        }
        for mul_triples.each |elm| {
            let (aVec, bVec, cVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);

            if a.is_not_zero() { check_quotrem(c, a, b, BigInt::zero()); }
            if b.is_not_zero() { check_quotrem(c, b, a, BigInt::zero()); }
        }

        for divmod_quadruples.each |elm| {
            let (aVec, bVec, cVec, dVec) = *elm;
            let a = BigInt::from_slice(Plus, aVec);
            let b = BigInt::from_slice(Plus, bVec);
            let c = BigInt::from_slice(Plus, cVec);
            let d = BigInt::from_slice(Plus, dVec);

            if b.is_not_zero() {
                check_quotrem(a, b, c, d);
            }
        }
    }

    #[test]
    fn test_to_str_radix() {
        assert BigInt::from_biguint(Plus, BigUint::from_uint(10))
            .to_str_radix(10) == ~"10";
        assert BigInt::one().to_str_radix(10) == ~"1";
        assert BigInt::zero().to_str_radix(10) == ~"0";
        assert (-BigInt::one()).to_str_radix(10) == ~"-1";
        assert BigInt::from_biguint(Minus, BigUint::from_uint(10))
            .to_str_radix(10) == ~"-10";
    }


    #[test]
    fn test_from_str_radix() {
        assert BigInt::from_biguint(Plus, BigUint::from_uint(10)) ==
            BigInt::from_str_radix(~"10", 10).get();
        assert BigInt::one()== BigInt::from_str_radix(~"1", 10).get();
        assert BigInt::zero() == BigInt::from_str_radix(~"0", 10).get();
        assert (-BigInt::one()) == BigInt::from_str_radix(~"-1", 10).get();
        assert BigInt::from_biguint(Minus, BigUint::from_uint(10)) ==
            BigInt::from_str_radix(~"-10", 10).get();

        assert BigInt::from_str_radix(~"Z", 10) == None;
        assert BigInt::from_str_radix(~"_", 2) == None;
        assert BigInt::from_str_radix(~"-1", 10) ==
            Some(BigInt::from_biguint(Minus, BigUint::one()));
    }

    #[test]
    fn test_neg() {
        assert -BigInt::from_at_vec(Plus,  @[1, 1, 1]) ==
            BigInt::from_at_vec(Minus, @[1, 1, 1]);
        assert -BigInt::from_at_vec(Minus, @[1, 1, 1]) ==
            BigInt::from_at_vec(Plus,  @[1, 1, 1]);
        assert -BigInt::zero() == BigInt::zero();
    }
}

