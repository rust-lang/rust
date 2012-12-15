/*!

A Big signed integer.

A BigInt is a combination of BigUint and Sign.
*/

use core::cmp::{Eq, Ord};

use biguint::{BigDigit, BigUint};

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
    pure fn cmp(other: &Sign) -> int {
        match (self, *other) {
          (Minus, Minus) | (Zero,  Zero) | (Plus, Plus) =>  0,
          (Minus, Zero)  | (Minus, Plus) | (Zero, Plus) => -1,
          _                                             =>  1
        }
    }

    /// Negate Sign value.
    pure fn neg() -> Sign {
        match(self) {
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

    pure fn abs() -> BigInt { BigInt::from_biguint(Plus, self.data) }

    pure fn cmp(other: &BigInt) -> int {
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

    pure fn divmod(other: &BigInt) -> (BigInt, BigInt) {
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

    pure fn quot(other: &BigInt) -> BigInt { self.quotrem(other).first() }
    pure fn rem(other: &BigInt) -> BigInt { self.quotrem(other).second() }

    pure fn quotrem(other: &BigInt) -> (BigInt, BigInt) {
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

    pure fn is_zero() -> bool { self.sign == Zero }
    pure fn is_not_zero() -> bool { self.sign != Zero }
    pure fn is_positive() -> bool { self.sign == Plus }
    pure fn is_negative() -> bool { self.sign == Minus }
    pure fn is_nonpositive() -> bool { self.sign != Plus }
    pure fn is_nonnegative() -> bool { self.sign != Minus }

    pure fn to_uint() -> uint {
        match self.sign {
            Plus  => self.data.to_uint(),
            Zero  => 0,
            Minus => 0
        }
    }

    pure fn to_str_radix(radix: uint) -> ~str {
        match self.sign {
            Plus  => self.data.to_str_radix(radix),
            Zero  => ~"0",
            Minus => ~"-" + self.data.to_str_radix(radix)
        }
    }
}

#[cfg(test)]
mod tests {
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
            assert b == num::from_int(i);
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

