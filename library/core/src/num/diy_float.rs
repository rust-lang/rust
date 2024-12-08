//! Extended precision "soft float", for internal use only.

// This module is only for dec2flt and flt2dec, and only public because of coretests.
// It is not intended to ever be stabilized.
#![doc(hidden)]
#![unstable(
    feature = "core_private_diy_float",
    reason = "internal routines only exposed for testing",
    issue = "none"
)]

/// A custom 64-bit floating point type, representing `f * 2^e`.
#[derive(Copy, Clone, Debug)]
#[doc(hidden)]
pub struct Fp {
    /// The integer mantissa.
    pub f: u64,
    /// The exponent in base 2.
    pub e: i16,
}

impl Fp {
    /// Returns a correctly rounded product of itself and `other`.
    pub fn mul(&self, other: &Fp) -> Fp {
        const MASK: u64 = 0xffffffff;
        let a = self.f >> 32;
        let b = self.f & MASK;
        let c = other.f >> 32;
        let d = other.f & MASK;
        let ac = a * c;
        let bc = b * c;
        let ad = a * d;
        let bd = b * d;
        let tmp = (bd >> 32) + (ad & MASK) + (bc & MASK) + (1 << 31) /* round */;
        let f = ac + (ad >> 32) + (bc >> 32) + (tmp >> 32);
        let e = self.e + other.e + 64;
        Fp { f, e }
    }

    /// Normalizes itself so that the resulting mantissa is at least `2^63`.
    pub fn normalize(&self) -> Fp {
        let mut f = self.f;
        let mut e = self.e;
        if f >> (64 - 32) == 0 {
            f <<= 32;
            e -= 32;
        }
        if f >> (64 - 16) == 0 {
            f <<= 16;
            e -= 16;
        }
        if f >> (64 - 8) == 0 {
            f <<= 8;
            e -= 8;
        }
        if f >> (64 - 4) == 0 {
            f <<= 4;
            e -= 4;
        }
        if f >> (64 - 2) == 0 {
            f <<= 2;
            e -= 2;
        }
        if f >> (64 - 1) == 0 {
            f <<= 1;
            e -= 1;
        }
        debug_assert!(f >= (1 << 63));
        Fp { f, e }
    }

    /// Normalizes itself to have the shared exponent.
    /// It can only decrease the exponent (and thus increase the mantissa).
    pub fn normalize_to(&self, e: i16) -> Fp {
        let edelta = self.e - e;
        assert!(edelta >= 0);
        let edelta = edelta as usize;
        assert_eq!(self.f << edelta >> edelta, self.f);
        Fp { f: self.f << edelta, e }
    }
}
