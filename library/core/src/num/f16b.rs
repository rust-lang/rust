//! The 16-bit brain floating-point type.

#![unstable(feature = "f16b", issue = "160630")]

use crate::{fmt, mem};

/// A 16-bit brain floating-point value.
///
/// This type stores values using the bfloat16 encoding. It deliberately
/// exposes only raw-bit construction, comparison, formatting, and lossless
/// widening to [`f32`].
///
/// The 16-bit brain floating-point intends to preserve the dynamic range of
/// a 32-bit floating-point value while using half the storage. It does
/// this by using 8 bits for the exponent, the same as `f32`, but only
/// using 7 bits for the mantissa. See [Wikipedia on bfloat16][wikipedia] for
/// more information.
///
/// [wikipedia]: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
#[lang = "f16b"]
#[doc(alias = "bf16")] // what hardware often names it
#[doc(alias = "bfloat")] // LLVM's name
#[doc(alias = "bfloat16")] // Wikipedia's name
#[doc(alias = "bfloat16_t")] // The C++ `stdfloat` name
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[unstable(feature = "f16b", issue = "160630")]
pub struct f16b(u16);

#[doc(test(attr(
    feature(cfg_target_has_reliable_f16b),
    allow(internal_features, unused_features)
)))]
impl f16b {
    /// Raw transmutation from `u16`.
    ///
    /// This is currently identical to `transmute::<u16, f16b>(v)` on all platforms.
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
    ///
    /// ```no_run
    /// #![feature(f16b)]
    /// # #[cfg(target_has_reliable_f16b)] {
    /// # use core::num::f16b;
    ///
    /// let v = f16b::from_bits(0x4148);
    /// assert_eq!(f32::from(v), 12.5);
    /// # }
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "f16b", issue = "160630")]
    pub const fn from_bits(bits: u16) -> Self {
        // SAFETY: `f16b` and `u16` have the same size, and every bit pattern is valid.
        unsafe { mem::transmute(bits) }
    }

    /// Raw transmutation to `u16`.
    ///
    /// This is currently identical to `transmute::<f16b, u16>(self)` on all platforms.
    ///
    /// See [`from_bits`](#method.from_bits) for some discussion of the
    /// portability of this operation (there are almost no issues).
    ///
    /// Note that this function is distinct from `as` casting, which attempts to
    /// preserve the *numeric* value, and not the bitwise value.
    ///
    /// ```no_run
    /// #![feature(f16b)]
    /// # #[cfg(target_has_reliable_f16b)] {
    /// # use core::num::f16b;
    ///
    /// assert_eq!(f16b::from_bits(0x4148).to_bits(), 0x4148);
    /// # }
    /// ```
    #[inline]
    #[unstable(feature = "f16b", issue = "160630")]
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub const fn to_bits(self) -> u16 {
        // SAFETY: `f16b` and `u16` have the same size, and every bit pattern is valid.
        unsafe { mem::transmute(self) }
    }
}

#[inline]
const fn widen(value: f16b) -> f32 {
    f32::from_bits((value.to_bits() as u32) << 16)
}

#[unstable(feature = "f16b", issue = "160630")]
impl Copy for f16b {}

#[unstable(feature = "f16b", issue = "160630")]
impl Clone for f16b {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

#[unstable(feature = "f16b", issue = "160630")]
impl Default for f16b {
    #[inline]
    fn default() -> Self {
        Self::from_bits(0)
    }
}

#[unstable(feature = "f16b", issue = "160630")]
impl PartialEq for f16b {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        widen(*self).eq(&widen(*other))
    }
}

#[unstable(feature = "f16b", issue = "160630")]
impl PartialOrd for f16b {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<crate::cmp::Ordering> {
        widen(*self).partial_cmp(&widen(*other))
    }
}

#[unstable(feature = "f16b", issue = "160630")]
impl From<f16b> for f32 {
    #[inline]
    fn from(value: f16b) -> Self {
        widen(value)
    }
}

#[unstable(feature = "f16b", issue = "160630")]
impl fmt::Debug for f16b {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&widen(*self), formatter)
    }
}

#[cfg(not(no_fp_fmt_parse))]
#[unstable(feature = "f16b", issue = "160630")]
impl fmt::LowerExp for f16b {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerExp::fmt(&widen(*self), formatter)
    }
}

#[cfg(not(no_fp_fmt_parse))]
#[unstable(feature = "f16b", issue = "160630")]
impl fmt::UpperExp for f16b {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperExp::fmt(&widen(*self), formatter)
    }
}
