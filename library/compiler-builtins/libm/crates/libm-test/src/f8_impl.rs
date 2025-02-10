//! An IEEE-compliant 8-bit float type for testing purposes.

use std::cmp::{self, Ordering};
use std::{fmt, ops};

use libm::support::hex_float::parse_any;

use crate::Float;

/// Sometimes verifying float logic is easiest when all values can quickly be checked exhaustively
/// or by hand.
///
/// IEEE-754 compliant type that includes a 1 bit sign, 4 bit exponent, and 3 bit significand.
/// Bias is -7.
///
/// Based on <https://en.wikipedia.org/wiki/Minifloat#Example_8-bit_float_(1.4.3)>.
#[derive(Clone, Copy)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub struct f8(u8);

impl Float for f8 {
    type Int = u8;
    type SignedInt = i8;

    const ZERO: Self = Self(0b0_0000_000);
    const NEG_ZERO: Self = Self(0b1_0000_000);
    const ONE: Self = Self(0b0_0111_000);
    const NEG_ONE: Self = Self(0b1_0111_000);
    const MAX: Self = Self(0b0_1110_111);
    const MIN: Self = Self(0b1_1110_111);
    const INFINITY: Self = Self(0b0_1111_000);
    const NEG_INFINITY: Self = Self(0b1_1111_000);
    const NAN: Self = Self(0b0_1111_100);
    const MIN_POSITIVE_NORMAL: Self = Self(1 << Self::SIG_BITS);
    // FIXME: incorrect values
    const EPSILON: Self = Self::ZERO;
    const PI: Self = Self::ZERO;
    const NEG_PI: Self = Self::ZERO;
    const FRAC_PI_2: Self = Self::ZERO;

    const BITS: u32 = 8;
    const SIG_BITS: u32 = 3;
    const SIGN_MASK: Self::Int = 0b1_0000_000;
    const SIG_MASK: Self::Int = 0b0_0000_111;
    const EXP_MASK: Self::Int = 0b0_1111_000;
    const IMPLICIT_BIT: Self::Int = 0b0_0001_000;

    fn to_bits(self) -> Self::Int {
        self.0
    }

    fn to_bits_signed(self) -> Self::SignedInt {
        self.0 as i8
    }

    fn is_nan(self) -> bool {
        self.0 & Self::EXP_MASK == Self::EXP_MASK && self.0 & Self::SIG_MASK != 0
    }

    fn is_infinite(self) -> bool {
        self.0 & Self::EXP_MASK == Self::EXP_MASK && self.0 & Self::SIG_MASK == 0
    }

    fn is_sign_negative(self) -> bool {
        self.0 & Self::SIGN_MASK != 0
    }

    fn from_bits(a: Self::Int) -> Self {
        Self(a)
    }

    fn abs(self) -> Self {
        libm::generic::fabs(self)
    }

    fn copysign(self, other: Self) -> Self {
        libm::generic::copysign(self, other)
    }

    fn normalize(_significand: Self::Int) -> (i32, Self::Int) {
        unimplemented!()
    }
}

impl f8 {
    pub const ALL_LEN: usize = 240;

    /// All non-infinite non-NaN values of `f8`
    pub const ALL: [Self; Self::ALL_LEN] = [
        // -m*2^7
        Self(0b1_1110_111), // -240
        Self(0b1_1110_110),
        Self(0b1_1110_101),
        Self(0b1_1110_100),
        Self(0b1_1110_011),
        Self(0b1_1110_010),
        Self(0b1_1110_001),
        Self(0b1_1110_000), // -128
        // -m*2^6
        Self(0b1_1101_111), // -120
        Self(0b1_1101_110),
        Self(0b1_1101_101),
        Self(0b1_1101_100),
        Self(0b1_1101_011),
        Self(0b1_1101_010),
        Self(0b1_1101_001),
        Self(0b1_1101_000), // -64
        // -m*2^5
        Self(0b1_1100_111), // -60
        Self(0b1_1100_110),
        Self(0b1_1100_101),
        Self(0b1_1100_100),
        Self(0b1_1100_011),
        Self(0b1_1100_010),
        Self(0b1_1100_001),
        Self(0b1_1100_000), // -32
        // -m*2^4
        Self(0b1_1011_111), // -30
        Self(0b1_1011_110),
        Self(0b1_1011_101),
        Self(0b1_1011_100),
        Self(0b1_1011_011),
        Self(0b1_1011_010),
        Self(0b1_1011_001),
        Self(0b1_1011_000), // -16
        // -m*2^3
        Self(0b1_1010_111), // -15
        Self(0b1_1010_110),
        Self(0b1_1010_101),
        Self(0b1_1010_100),
        Self(0b1_1010_011),
        Self(0b1_1010_010),
        Self(0b1_1010_001),
        Self(0b1_1010_000), // -8
        // -m*2^2
        Self(0b1_1001_111), // -7.5
        Self(0b1_1001_110),
        Self(0b1_1001_101),
        Self(0b1_1001_100),
        Self(0b1_1001_011),
        Self(0b1_1001_010),
        Self(0b1_1001_001),
        Self(0b1_1001_000), // -4
        // -m*2^1
        Self(0b1_1000_111), // -3.75
        Self(0b1_1000_110),
        Self(0b1_1000_101),
        Self(0b1_1000_100),
        Self(0b1_1000_011),
        Self(0b1_1000_010),
        Self(0b1_1000_001),
        Self(0b1_1000_000), // -2
        // -m*2^0
        Self(0b1_0111_111), // -1.875
        Self(0b1_0111_110),
        Self(0b1_0111_101),
        Self(0b1_0111_100),
        Self(0b1_0111_011),
        Self(0b1_0111_010),
        Self(0b1_0111_001),
        Self(0b1_0111_000), // -1
        // -m*2^-1
        Self(0b1_0110_111), // −0.9375
        Self(0b1_0110_110),
        Self(0b1_0110_101),
        Self(0b1_0110_100),
        Self(0b1_0110_011),
        Self(0b1_0110_010),
        Self(0b1_0110_001),
        Self(0b1_0110_000), // -0.5
        // -m*2^-2
        Self(0b1_0101_111), // −0.46875
        Self(0b1_0101_110),
        Self(0b1_0101_101),
        Self(0b1_0101_100),
        Self(0b1_0101_011),
        Self(0b1_0101_010),
        Self(0b1_0101_001),
        Self(0b1_0101_000), // -0.25
        // -m*2^-3
        Self(0b1_0100_111), // −0.234375
        Self(0b1_0100_110),
        Self(0b1_0100_101),
        Self(0b1_0100_100),
        Self(0b1_0100_011),
        Self(0b1_0100_010),
        Self(0b1_0100_001),
        Self(0b1_0100_000), // -0.125
        // -m*2^-4
        Self(0b1_0011_111), // −0.1171875
        Self(0b1_0011_110),
        Self(0b1_0011_101),
        Self(0b1_0011_100),
        Self(0b1_0011_011),
        Self(0b1_0011_010),
        Self(0b1_0011_001),
        Self(0b1_0011_000), // −0.0625
        // -m*2^-5
        Self(0b1_0010_111), // −0.05859375
        Self(0b1_0010_110),
        Self(0b1_0010_101),
        Self(0b1_0010_100),
        Self(0b1_0010_011),
        Self(0b1_0010_010),
        Self(0b1_0010_001),
        Self(0b1_0010_000), // −0.03125
        // -m*2^-6
        Self(0b1_0001_111), // −0.029296875
        Self(0b1_0001_110),
        Self(0b1_0001_101),
        Self(0b1_0001_100),
        Self(0b1_0001_011),
        Self(0b1_0001_010),
        Self(0b1_0001_001),
        Self(0b1_0001_000), // −0.015625
        // -m*2^-7 subnormal numbers
        Self(0b1_0000_111), // −0.013671875
        Self(0b1_0000_110),
        Self(0b1_0000_101),
        Self(0b1_0000_100),
        Self(0b1_0000_011),
        Self(0b1_0000_010),
        Self(0b1_0000_001), // −0.001953125
        // Zeroes
        Self(0b1_0000_000), // -0.0
        Self(0b0_0000_000), // 0.0
        // m*2^-7 // subnormal numbers
        Self(0b0_0000_001),
        Self(0b0_0000_010),
        Self(0b0_0000_011),
        Self(0b0_0000_100),
        Self(0b0_0000_101),
        Self(0b0_0000_110),
        Self(0b0_0000_111), // 0.013671875
        // m*2^-6
        Self(0b0_0001_000), // 0.015625
        Self(0b0_0001_001),
        Self(0b0_0001_010),
        Self(0b0_0001_011),
        Self(0b0_0001_100),
        Self(0b0_0001_101),
        Self(0b0_0001_110),
        Self(0b0_0001_111), // 0.029296875
        // m*2^-5
        Self(0b0_0010_000), // 0.03125
        Self(0b0_0010_001),
        Self(0b0_0010_010),
        Self(0b0_0010_011),
        Self(0b0_0010_100),
        Self(0b0_0010_101),
        Self(0b0_0010_110),
        Self(0b0_0010_111), // 0.05859375
        // m*2^-4
        Self(0b0_0011_000), // 0.0625
        Self(0b0_0011_001),
        Self(0b0_0011_010),
        Self(0b0_0011_011),
        Self(0b0_0011_100),
        Self(0b0_0011_101),
        Self(0b0_0011_110),
        Self(0b0_0011_111), // 0.1171875
        // m*2^-3
        Self(0b0_0100_000), // 0.125
        Self(0b0_0100_001),
        Self(0b0_0100_010),
        Self(0b0_0100_011),
        Self(0b0_0100_100),
        Self(0b0_0100_101),
        Self(0b0_0100_110),
        Self(0b0_0100_111), // 0.234375
        // m*2^-2
        Self(0b0_0101_000), // 0.25
        Self(0b0_0101_001),
        Self(0b0_0101_010),
        Self(0b0_0101_011),
        Self(0b0_0101_100),
        Self(0b0_0101_101),
        Self(0b0_0101_110),
        Self(0b0_0101_111), // 0.46875
        // m*2^-1
        Self(0b0_0110_000), // 0.5
        Self(0b0_0110_001),
        Self(0b0_0110_010),
        Self(0b0_0110_011),
        Self(0b0_0110_100),
        Self(0b0_0110_101),
        Self(0b0_0110_110),
        Self(0b0_0110_111), // 0.9375
        // m*2^0
        Self(0b0_0111_000), // 1
        Self(0b0_0111_001),
        Self(0b0_0111_010),
        Self(0b0_0111_011),
        Self(0b0_0111_100),
        Self(0b0_0111_101),
        Self(0b0_0111_110),
        Self(0b0_0111_111), // 1.875
        // m*2^1
        Self(0b0_1000_000), // 2
        Self(0b0_1000_001),
        Self(0b0_1000_010),
        Self(0b0_1000_011),
        Self(0b0_1000_100),
        Self(0b0_1000_101),
        Self(0b0_1000_110),
        Self(0b0_1000_111), // 3.75
        // m*2^2
        Self(0b0_1001_000), // 4
        Self(0b0_1001_001),
        Self(0b0_1001_010),
        Self(0b0_1001_011),
        Self(0b0_1001_100),
        Self(0b0_1001_101),
        Self(0b0_1001_110),
        Self(0b0_1001_111), // 7.5
        // m*2^3
        Self(0b0_1010_000), // 8
        Self(0b0_1010_001),
        Self(0b0_1010_010),
        Self(0b0_1010_011),
        Self(0b0_1010_100),
        Self(0b0_1010_101),
        Self(0b0_1010_110),
        Self(0b0_1010_111), // 15
        // m*2^4
        Self(0b0_1011_000), // 16
        Self(0b0_1011_001),
        Self(0b0_1011_010),
        Self(0b0_1011_011),
        Self(0b0_1011_100),
        Self(0b0_1011_101),
        Self(0b0_1011_110),
        Self(0b0_1011_111), // 30
        // m*2^5
        Self(0b0_1100_000), // 32
        Self(0b0_1100_001),
        Self(0b0_1100_010),
        Self(0b0_1100_011),
        Self(0b0_1100_100),
        Self(0b0_1100_101),
        Self(0b0_1100_110),
        Self(0b0_1100_111), // 60
        // m*2^6
        Self(0b0_1101_000), // 64
        Self(0b0_1101_001),
        Self(0b0_1101_010),
        Self(0b0_1101_011),
        Self(0b0_1101_100),
        Self(0b0_1101_101),
        Self(0b0_1101_110),
        Self(0b0_1101_111), // 120
        // m*2^7
        Self(0b0_1110_000), // 128
        Self(0b0_1110_001),
        Self(0b0_1110_010),
        Self(0b0_1110_011),
        Self(0b0_1110_100),
        Self(0b0_1110_101),
        Self(0b0_1110_110),
        Self(0b0_1110_111), // 240
    ];
}

impl ops::Add for f8 {
    type Output = Self;
    fn add(self, _rhs: Self) -> Self::Output {
        unimplemented!()
    }
}

impl ops::Sub for f8 {
    type Output = Self;
    fn sub(self, _rhs: Self) -> Self::Output {
        unimplemented!()
    }
}
impl ops::Mul for f8 {
    type Output = Self;
    fn mul(self, _rhs: Self) -> Self::Output {
        unimplemented!()
    }
}
impl ops::Div for f8 {
    type Output = Self;
    fn div(self, _rhs: Self) -> Self::Output {
        unimplemented!()
    }
}

impl ops::Neg for f8 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(self.0 ^ Self::SIGN_MASK)
    }
}

impl ops::Rem for f8 {
    type Output = Self;
    fn rem(self, _rhs: Self) -> Self::Output {
        unimplemented!()
    }
}

impl ops::AddAssign for f8 {
    fn add_assign(&mut self, _rhs: Self) {
        unimplemented!()
    }
}

impl ops::SubAssign for f8 {
    fn sub_assign(&mut self, _rhs: Self) {
        unimplemented!()
    }
}

impl ops::MulAssign for f8 {
    fn mul_assign(&mut self, _rhs: Self) {
        unimplemented!()
    }
}

impl cmp::PartialEq for f8 {
    fn eq(&self, other: &Self) -> bool {
        if self.is_nan() || other.is_nan() {
            false
        } else if self.abs().to_bits() | other.abs().to_bits() == 0 {
            true
        } else {
            self.0 == other.0
        }
    }
}
impl cmp::PartialOrd for f8 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let inf_rep = f8::EXP_MASK;

        let a_abs = self.abs().to_bits();
        let b_abs = other.abs().to_bits();

        // If either a or b is NaN, they are unordered.
        if a_abs > inf_rep || b_abs > inf_rep {
            return None;
        }

        // If a and b are both zeros, they are equal.
        if a_abs | b_abs == 0 {
            return Some(Ordering::Equal);
        }

        let a_srep = self.to_bits_signed();
        let b_srep = other.to_bits_signed();
        let res = a_srep.cmp(&b_srep);

        if a_srep & b_srep >= 0 {
            // If at least one of a and b is positive, we get the same result comparing
            // a and b as signed integers as we would with a fp_ting-point compare.
            Some(res)
        } else {
            // Otherwise, both are negative, so we need to flip the sense of the
            // comparison to get the correct result.
            Some(res.reverse())
        }
    }
}
impl fmt::Display for f8 {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unimplemented!()
    }
}

impl fmt::Debug for f8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Binary::fmt(self, f)
    }
}

impl fmt::Binary for f8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.0;
        write!(
            f,
            "0b{:b}_{:04b}_{:03b}",
            v >> 7,
            (v & Self::EXP_MASK) >> Self::SIG_BITS,
            v & Self::SIG_MASK
        )
    }
}

impl fmt::LowerHex for f8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

pub const fn hf8(s: &str) -> f8 {
    f8(parse_any(s, 8, 3) as u8)
}
