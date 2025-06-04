//! Integers used for wide operations, larger than `u128`.

#![allow(unused)]

use core::{fmt, ops};

use crate::int::{DInt, HInt, Int, MinInt};

const WORD_LO_MASK: u64 = 0x00000000ffffffff;
const WORD_HI_MASK: u64 = 0xffffffff00000000;
const WORD_FULL_MASK: u64 = 0xffffffffffffffff;
const U128_LO_MASK: u128 = u64::MAX as u128;
const U128_HI_MASK: u128 = (u64::MAX as u128) << 64;

/// A 256-bit unsigned integer represented as 4 64-bit limbs.
///
/// Each limb is a native-endian number, but the array is little-limb-endian.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct u256(pub [u64; 4]);

impl u256 {
    pub const MAX: Self = Self([u64::MAX, u64::MAX, u64::MAX, u64::MAX]);

    /// Reinterpret as a signed integer
    pub fn signed(self) -> i256 {
        i256(self.0)
    }
}

/// A 256-bit signed integer represented as 4 64-bit limbs.
///
/// Each limb is a native-endian number, but the array is little-limb-endian.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct i256(pub [u64; 4]);

impl i256 {
    /// Reinterpret as an unsigned integer
    pub fn unsigned(self) -> u256 {
        u256(self.0)
    }
}

impl MinInt for u256 {
    type OtherSign = i256;

    type UnsignedInt = u256;

    const SIGNED: bool = false;
    const BITS: u32 = 256;
    const ZERO: Self = Self([0u64; 4]);
    const ONE: Self = Self([1, 0, 0, 0]);
    const MIN: Self = Self([0u64; 4]);
    const MAX: Self = Self([u64::MAX; 4]);
}

impl MinInt for i256 {
    type OtherSign = u256;

    type UnsignedInt = u256;

    const SIGNED: bool = false;
    const BITS: u32 = 256;
    const ZERO: Self = Self([0u64; 4]);
    const ONE: Self = Self([1, 0, 0, 0]);
    const MIN: Self = Self([0, 0, 0, 1 << 63]);
    const MAX: Self = Self([u64::MAX, u64::MAX, u64::MAX, u64::MAX >> 1]);
}

macro_rules! impl_common {
    ($ty:ty) => {
        impl ops::BitOr for $ty {
            type Output = Self;

            fn bitor(mut self, rhs: Self) -> Self::Output {
                self.0[0] |= rhs.0[0];
                self.0[1] |= rhs.0[1];
                self.0[2] |= rhs.0[2];
                self.0[3] |= rhs.0[3];
                self
            }
        }

        impl ops::Not for $ty {
            type Output = Self;

            fn not(self) -> Self::Output {
                Self([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
            }
        }

        impl ops::Shl<u32> for $ty {
            type Output = Self;

            fn shl(self, rhs: u32) -> Self::Output {
                unimplemented!("only used to meet trait bounds")
            }
        }
    };
}

impl_common!(i256);
impl_common!(u256);

impl ops::Shr<u32> for u256 {
    type Output = Self;

    fn shr(self, rhs: u32) -> Self::Output {
        assert!(rhs < Self::BITS, "attempted to shift right with overflow");

        if rhs == 0 {
            return self;
        }

        let mut ret = self;
        let byte_shift = rhs / 64;
        let bit_shift = rhs % 64;

        for idx in 0..4 {
            let base_idx = idx + byte_shift as usize;

            let Some(base) = ret.0.get(base_idx) else {
                ret.0[idx] = 0;
                continue;
            };

            let mut new_val = base >> bit_shift;

            if let Some(new) = ret.0.get(base_idx + 1) {
                new_val |= new.overflowing_shl(64 - bit_shift).0;
            }

            ret.0[idx] = new_val;
        }

        ret
    }
}

macro_rules! word {
    (1, $val:expr) => {
        (($val >> (32 * 3)) & Self::from(WORD_LO_MASK)) as u64
    };
    (2, $val:expr) => {
        (($val >> (32 * 2)) & Self::from(WORD_LO_MASK)) as u64
    };
    (3, $val:expr) => {
        (($val >> (32 * 1)) & Self::from(WORD_LO_MASK)) as u64
    };
    (4, $val:expr) => {
        (($val >> (32 * 0)) & Self::from(WORD_LO_MASK)) as u64
    };
}

impl HInt for u128 {
    type D = u256;

    fn widen(self) -> Self::D {
        let w0 = self & u128::from(u64::MAX);
        let w1 = (self >> u64::BITS) & u128::from(u64::MAX);
        u256([w0 as u64, w1 as u64, 0, 0])
    }

    fn zero_widen(self) -> Self::D {
        self.widen()
    }

    fn zero_widen_mul(self, rhs: Self) -> Self::D {
        let product11: u64 = word!(1, self) * word!(1, rhs);
        let product12: u64 = word!(1, self) * word!(2, rhs);
        let product13: u64 = word!(1, self) * word!(3, rhs);
        let product14: u64 = word!(1, self) * word!(4, rhs);
        let product21: u64 = word!(2, self) * word!(1, rhs);
        let product22: u64 = word!(2, self) * word!(2, rhs);
        let product23: u64 = word!(2, self) * word!(3, rhs);
        let product24: u64 = word!(2, self) * word!(4, rhs);
        let product31: u64 = word!(3, self) * word!(1, rhs);
        let product32: u64 = word!(3, self) * word!(2, rhs);
        let product33: u64 = word!(3, self) * word!(3, rhs);
        let product34: u64 = word!(3, self) * word!(4, rhs);
        let product41: u64 = word!(4, self) * word!(1, rhs);
        let product42: u64 = word!(4, self) * word!(2, rhs);
        let product43: u64 = word!(4, self) * word!(3, rhs);
        let product44: u64 = word!(4, self) * word!(4, rhs);

        let sum0: u128 = u128::from(product44);
        let sum1: u128 = u128::from(product34) + u128::from(product43);
        let sum2: u128 = u128::from(product24) + u128::from(product33) + u128::from(product42);
        let sum3: u128 = u128::from(product14)
            + u128::from(product23)
            + u128::from(product32)
            + u128::from(product41);
        let sum4: u128 = u128::from(product13) + u128::from(product22) + u128::from(product31);
        let sum5: u128 = u128::from(product12) + u128::from(product21);
        let sum6: u128 = u128::from(product11);

        let r0: u128 =
            (sum0 & u128::from(WORD_FULL_MASK)) + ((sum1 & u128::from(WORD_LO_MASK)) << 32);
        let r1: u128 = (sum0 >> 64)
            + ((sum1 >> 32) & u128::from(WORD_FULL_MASK))
            + (sum2 & u128::from(WORD_FULL_MASK))
            + ((sum3 << 32) & u128::from(WORD_HI_MASK));

        let (lo, carry) = r0.overflowing_add(r1 << 64);
        let hi = (r1 >> 64)
            + (sum1 >> 96)
            + (sum2 >> 64)
            + (sum3 >> 32)
            + sum4
            + (sum5 << 32)
            + (sum6 << 64)
            + u128::from(carry);

        u256([
            (lo & U128_LO_MASK) as u64,
            ((lo >> 64) & U128_LO_MASK) as u64,
            (hi & U128_LO_MASK) as u64,
            ((hi >> 64) & U128_LO_MASK) as u64,
        ])
    }

    fn widen_mul(self, rhs: Self) -> Self::D {
        self.zero_widen_mul(rhs)
    }

    fn widen_hi(self) -> Self::D {
        self.widen() << <Self as MinInt>::BITS
    }
}

impl HInt for i128 {
    type D = i256;

    fn widen(self) -> Self::D {
        let mut ret = self.unsigned().zero_widen().signed();
        if self.is_negative() {
            ret.0[2] = u64::MAX;
            ret.0[3] = u64::MAX;
        }
        ret
    }

    fn zero_widen(self) -> Self::D {
        self.unsigned().zero_widen().signed()
    }

    fn zero_widen_mul(self, rhs: Self) -> Self::D {
        self.unsigned().zero_widen_mul(rhs.unsigned()).signed()
    }

    fn widen_mul(self, rhs: Self) -> Self::D {
        unimplemented!("signed i128 widening multiply is not used")
    }

    fn widen_hi(self) -> Self::D {
        self.widen() << <Self as MinInt>::BITS
    }
}

impl DInt for u256 {
    type H = u128;

    fn lo(self) -> Self::H {
        let mut tmp = [0u8; 16];
        tmp[..8].copy_from_slice(&self.0[0].to_le_bytes());
        tmp[8..].copy_from_slice(&self.0[1].to_le_bytes());
        u128::from_le_bytes(tmp)
    }

    fn hi(self) -> Self::H {
        let mut tmp = [0u8; 16];
        tmp[..8].copy_from_slice(&self.0[2].to_le_bytes());
        tmp[8..].copy_from_slice(&self.0[3].to_le_bytes());
        u128::from_le_bytes(tmp)
    }
}

impl DInt for i256 {
    type H = i128;

    fn lo(self) -> Self::H {
        let mut tmp = [0u8; 16];
        tmp[..8].copy_from_slice(&self.0[0].to_le_bytes());
        tmp[8..].copy_from_slice(&self.0[1].to_le_bytes());
        i128::from_le_bytes(tmp)
    }

    fn hi(self) -> Self::H {
        let mut tmp = [0u8; 16];
        tmp[..8].copy_from_slice(&self.0[2].to_le_bytes());
        tmp[8..].copy_from_slice(&self.0[3].to_le_bytes());
        i128::from_le_bytes(tmp)
    }
}
