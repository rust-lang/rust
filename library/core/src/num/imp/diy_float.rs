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
    pub fn mul(self, other: Self) -> Self {
        let (lo, hi) = self.f.widening_mul(other.f);
        let f = hi + (lo >> 63) /* round */;
        let e = self.e + other.e + 64;
        Self { f, e }
    }

    /// Normalizes itself so that the resulting mantissa is at least `2^63`.
    pub fn normalize(self) -> Self {
        let lz = self.f.leading_zeros();
        let f = self.f << lz;
        let e = self.e - lz as i16;
        debug_assert!(f >= (1 << 63));
        Self { f, e }
    }

    /// Normalizes itself to have the shared exponent.
    /// It can only decrease the exponent (and thus increase the mantissa).
    pub fn normalize_to(self, e: i16) -> Self {
        let edelta = self.e - e;
        assert!(edelta >= 0);
        let edelta = edelta as usize;
        assert_eq!(self.f << edelta >> edelta, self.f);
        Self { f: self.f << edelta, e }
    }
}
