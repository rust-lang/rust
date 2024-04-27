//! Constants for the `f16` half-precision floating point type.
//!
//! *[See also the `f16` primitive type][f16].*
//!
//! Mathematically significant numbers are provided in the `consts` sub-module.
//!
//! For the constants defined directly in this module
//! (as distinct from those defined in the `consts` sub-module),
//! new code should instead use the associated constants
//! defined directly on the `f16` type.

#![unstable(feature = "f16", issue = "116909")]

use crate::mem;

/// Basic mathematical constants.
#[unstable(feature = "f16", issue = "116909")]
pub mod consts {}

#[cfg(not(test))]
impl f16 {
    // FIXME(f16_f128): almost everything in this `impl` is missing examples and a const
    // implementation. Add these once we can run code on all platforms and have f16/f128 in CTFE.

    /// Returns `true` if this value is NaN.
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    #[allow(clippy::eq_op)] // > if you intended to check if the operand is NaN, use `.is_nan()` instead :)
    pub const fn is_nan(self) -> bool {
        self != self
    }

    /// Returns `true` if `self` has a positive sign, including `+0.0`, NaNs with
    /// positive sign bit and positive infinity. Note that IEEE 754 doesn't assign any
    /// meaning to the sign bit in case of a NaN, and as Rust doesn't guarantee that
    /// the bit pattern of NaNs are conserved over arithmetic operations, the result of
    /// `is_sign_positive` on a NaN might produce an unexpected result in some cases.
    /// See [explanation of NaN as a special value](f32) for more info.
    #[inline]
    #[must_use]
    #[unstable(feature = "f128", issue = "116909")]
    pub fn is_sign_positive(self) -> bool {
        !self.is_sign_negative()
    }

    /// Returns `true` if `self` has a negative sign, including `-0.0`, NaNs with
    /// negative sign bit and negative infinity. Note that IEEE 754 doesn't assign any
    /// meaning to the sign bit in case of a NaN, and as Rust doesn't guarantee that
    /// the bit pattern of NaNs are conserved over arithmetic operations, the result of
    /// `is_sign_negative` on a NaN might produce an unexpected result in some cases.
    /// See [explanation of NaN as a special value](f32) for more info.
    #[inline]
    #[must_use]
    #[unstable(feature = "f128", issue = "116909")]
    pub fn is_sign_negative(self) -> bool {
        // IEEE754 says: isSignMinus(x) is true if and only if x has negative sign. isSignMinus
        // applies to zeros and NaNs as well.
        // SAFETY: This is just transmuting to get the sign bit, it's fine.
        (self.to_bits() & (1 << 15)) != 0
    }

    /// Raw transmutation to `u16`.
    ///
    /// This is currently identical to `transmute::<f16, u16>(self)` on all platforms.
    ///
    /// See [`from_bits`](#method.from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// Note that this function is distinct from `as` casting, which attempts to
    /// preserve the *numeric* value, and not the bitwise value.
    #[inline]
    #[unstable(feature = "f16", issue = "116909")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub fn to_bits(self) -> u16 {
        // SAFETY: `u16` is a plain old datatype so we can always... uh...
        // ...look, just pretend you forgot what you just read.
        // Stability concerns.
        unsafe { mem::transmute(self) }
    }

    /// Raw transmutation from `u16`.
    ///
    /// This is currently identical to `transmute::<u16, f16>(v)` on all platforms.
    /// It turns out this is incredibly portable, for two reasons:
    ///
    /// * Floats and Ints have the same endianness on all supported platforms.
    /// * IEEE 754 very precisely specifies the bit layout of floats.
    ///
    /// However there is one caveat: prior to the 2008 version of IEEE 754, how
    /// to interpret the NaN signaling bit wasn't actually specified. Most platforms
    /// (notably x86 and ARM) picked the interpretation that was ultimately
    /// standardized in 2008, but some didn't (notably MIPS). As a result, all
    /// signaling NaNs on MIPS are quiet NaNs on x86, and vice-versa.
    ///
    /// Rather than trying to preserve signaling-ness cross-platform, this
    /// implementation favors preserving the exact bits. This means that
    /// any payloads encoded in NaNs will be preserved even if the result of
    /// this method is sent over the network from an x86 machine to a MIPS one.
    ///
    /// If the results of this method are only manipulated by the same
    /// architecture that produced them, then there is no portability concern.
    ///
    /// If the input isn't NaN, then there is no portability concern.
    ///
    /// If you don't care about signalingness (very likely), then there is no
    /// portability concern.
    ///
    /// Note that this function is distinct from `as` casting, which attempts to
    /// preserve the *numeric* value, and not the bitwise value.
    #[inline]
    #[must_use]
    #[unstable(feature = "f16", issue = "116909")]
    pub fn from_bits(v: u16) -> Self {
        // SAFETY: `u16` is a plain old datatype so we can always... uh...
        // ...look, just pretend you forgot what you just read.
        // Stability concerns.
        unsafe { mem::transmute(v) }
    }
}
