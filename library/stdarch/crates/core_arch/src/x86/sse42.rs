//! Streaming SIMD Extensions 4.2 (SSE4.2)
//!
//! Extends SSE4.1 with STTNI (String and Text New Instructions).

#[cfg(test)]
use stdarch_test::assert_instr;

use crate::{
    core_arch::{simd::*, x86::*},
    intrinsics::simd::*,
};

/// String contains unsigned 8-bit characters *(Default)*
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_UBYTE_OPS: i32 = 0b0000_0000;
/// String contains unsigned 16-bit characters
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_UWORD_OPS: i32 = 0b0000_0001;
/// String contains signed 8-bit characters
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_SBYTE_OPS: i32 = 0b0000_0010;
/// String contains unsigned 16-bit characters
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_SWORD_OPS: i32 = 0b0000_0011;

/// For each character in `a`, find if it is in `b` *(Default)*
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_CMP_EQUAL_ANY: i32 = 0b0000_0000;
/// For each character in `a`, determine if
/// `b[0] <= c <= b[1] or b[1] <= c <= b[2]...`
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_CMP_RANGES: i32 = 0b0000_0100;
/// The strings defined by `a` and `b` are equal
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_CMP_EQUAL_EACH: i32 = 0b0000_1000;
/// Search for the defined substring in the target
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_CMP_EQUAL_ORDERED: i32 = 0b0000_1100;

/// Do not negate results *(Default)*
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_POSITIVE_POLARITY: i32 = 0b0000_0000;
/// Negates results
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_NEGATIVE_POLARITY: i32 = 0b0001_0000;
/// Do not negate results before the end of the string
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_MASKED_POSITIVE_POLARITY: i32 = 0b0010_0000;
/// Negates results only before the end of the string
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_MASKED_NEGATIVE_POLARITY: i32 = 0b0011_0000;

/// **Index only**: return the least significant bit *(Default)*
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_LEAST_SIGNIFICANT: i32 = 0b0000_0000;
/// **Index only**: return the most significant bit
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_MOST_SIGNIFICANT: i32 = 0b0100_0000;

/// **Mask only**: return the bit mask
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_BIT_MASK: i32 = 0b0000_0000;
/// **Mask only**: return the byte mask
#[stable(feature = "simd_x86", since = "1.27.0")]
pub const _SIDD_UNIT_MASK: i32 = 0b0100_0000;

/// Compares packed strings with implicit lengths in `a` and `b` using the
/// control in `IMM8`, and return the generated mask.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpistrm)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpistrm, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpistrm<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(pcmpistrm128(a.as_i8x16(), b.as_i8x16(), IMM8 as i8)) }
}

/// Compares packed strings with implicit lengths in `a` and `b` using the
/// control in `IMM8` and return the generated index. Similar to
/// [`_mm_cmpestri`] with the exception that [`_mm_cmpestri`] requires the
/// lengths of `a` and `b` to be explicitly specified.
///
/// # Control modes
///
/// The control specified by `IMM8` may be one or more of the following.
///
/// ## Data size and signedness
///
///  - [`_SIDD_UBYTE_OPS`] - Default
///  - [`_SIDD_UWORD_OPS`]
///  - [`_SIDD_SBYTE_OPS`]
///  - [`_SIDD_SWORD_OPS`]
///
/// ## Comparison options
///  - [`_SIDD_CMP_EQUAL_ANY`] - Default
///  - [`_SIDD_CMP_RANGES`]
///  - [`_SIDD_CMP_EQUAL_EACH`]
///  - [`_SIDD_CMP_EQUAL_ORDERED`]
///
/// ## Result polarity
///  - [`_SIDD_POSITIVE_POLARITY`] - Default
///  - [`_SIDD_NEGATIVE_POLARITY`]
///
/// ## Bit returned
///  - [`_SIDD_LEAST_SIGNIFICANT`] - Default
///  - [`_SIDD_MOST_SIGNIFICANT`]
///
/// # Examples
///
/// Finds a substring using [`_SIDD_CMP_EQUAL_ORDERED`]
///
/// ```
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// #     if is_x86_feature_detected!("sse4.2") {
/// #         #[target_feature(enable = "sse4.2")]
/// #         unsafe fn worker() {
/// let haystack = b"This is a long string of text data\r\n\tthat extends
/// multiple lines";
/// let needle = b"\r\n\t\0\0\0\0\0\0\0\0\0\0\0\0\0";
///
/// let a = _mm_loadu_si128(needle.as_ptr() as *const _);
/// let hop = 16;
/// let mut indexes = Vec::new();
///
/// // Chunk the haystack into 16 byte chunks and find
/// // the first "\r\n\t" in the chunk.
/// for (i, chunk) in haystack.chunks(hop).enumerate() {
///     let b = _mm_loadu_si128(chunk.as_ptr() as *const _);
///     let idx = _mm_cmpistri(a, b, _SIDD_CMP_EQUAL_ORDERED);
///     if idx != 16 {
///         indexes.push((idx as usize) + (i * hop));
///     }
/// }
/// assert_eq!(indexes, vec![34]);
/// #         }
/// #         unsafe { worker(); }
/// #     }
/// # }
/// ```
///
/// The `_mm_cmpistri` intrinsic may also be used to find the existence of
/// one or more of a given set of characters in the haystack.
///
/// ```
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// #     if is_x86_feature_detected!("sse4.2") {
/// #         #[target_feature(enable = "sse4.2")]
/// #         unsafe fn worker() {
/// // Ensure your input is 16 byte aligned
/// let password = b"hunter2\0\0\0\0\0\0\0\0\0";
/// let special_chars = b"!@#$%^&*()[]:;<>";
///
/// // Load the input
/// let a = _mm_loadu_si128(special_chars.as_ptr() as *const _);
/// let b = _mm_loadu_si128(password.as_ptr() as *const _);
///
/// // Use _SIDD_CMP_EQUAL_ANY to find the index of any bytes in b
/// let idx = _mm_cmpistri(a.into(), b.into(), _SIDD_CMP_EQUAL_ANY);
///
/// if idx < 16 {
///     println!("Congrats! Your password contains a special character");
///     # panic!("{:?} does not contain a special character", password);
/// } else {
///     println!("Your password should contain a special character");
/// }
/// #         }
/// #         unsafe { worker(); }
/// #     }
/// # }
/// ```
///
/// Finds the index of the first character in the haystack that is within a
/// range of characters.
///
/// ```
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// #     if is_x86_feature_detected!("sse4.2") {
/// #         #[target_feature(enable = "sse4.2")]
/// #         unsafe fn worker() {
/// # let b = b":;<=>?@[\\]^_`abc";
/// # let b = _mm_loadu_si128(b.as_ptr() as *const _);
///
/// // Specify the ranges of values to be searched for [A-Za-z0-9].
/// let a = b"AZaz09\0\0\0\0\0\0\0\0\0\0";
/// let a = _mm_loadu_si128(a.as_ptr() as *const _);
///
/// // Use _SIDD_CMP_RANGES to find the index of first byte in ranges.
/// // Which in this case will be the first alpha numeric byte found
/// // in the string.
/// let idx = _mm_cmpistri(a, b, _SIDD_CMP_RANGES);
///
/// if idx < 16 {
///     println!("Found an alpha numeric character");
///     # assert_eq!(idx, 13);
/// } else {
///     println!("Did not find an alpha numeric character");
/// }
/// #         }
/// #         unsafe { worker(); }
/// #     }
/// # }
/// ```
///
/// Working with 16-bit characters.
///
/// ```
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// #     if is_x86_feature_detected!("sse4.2") {
/// #         #[target_feature(enable = "sse4.2")]
/// #         unsafe fn worker() {
/// # let mut some_utf16_words = [0u16; 8];
/// # let mut more_utf16_words = [0u16; 8];
/// # '❤'.encode_utf16(&mut some_utf16_words);
/// # '𝕊'.encode_utf16(&mut more_utf16_words);
/// // Load the input
/// let a = _mm_loadu_si128(some_utf16_words.as_ptr() as *const _);
/// let b = _mm_loadu_si128(more_utf16_words.as_ptr() as *const _);
///
/// // Specify _SIDD_UWORD_OPS to compare words instead of bytes, and
/// // use _SIDD_CMP_EQUAL_EACH to compare the two strings.
/// let idx = _mm_cmpistri(a, b, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_EACH);
///
/// if idx == 0 {
///     println!("16-bit unicode strings were equal!");
///     # panic!("Strings should not be equal!")
/// } else {
///     println!("16-bit unicode strings were not equal!");
/// }
/// #         }
/// #         unsafe { worker(); }
/// #     }
/// # }
/// ```
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpistri)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpistri, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpistri<const IMM8: i32>(a: __m128i, b: __m128i) -> i32 {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { pcmpistri128(a.as_i8x16(), b.as_i8x16(), IMM8 as i8) }
}

/// Compares packed strings with implicit lengths in `a` and `b` using the
/// control in `IMM8`, and return `1` if any character in `b` was null.
/// and `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpistrz)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpistri, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpistrz<const IMM8: i32>(a: __m128i, b: __m128i) -> i32 {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { pcmpistriz128(a.as_i8x16(), b.as_i8x16(), IMM8 as i8) }
}

/// Compares packed strings with implicit lengths in `a` and `b` using the
/// control in `IMM8`, and return `1` if the resulting mask was non-zero,
/// and `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpistrc)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpistri, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpistrc<const IMM8: i32>(a: __m128i, b: __m128i) -> i32 {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { pcmpistric128(a.as_i8x16(), b.as_i8x16(), IMM8 as i8) }
}

/// Compares packed strings with implicit lengths in `a` and `b` using the
/// control in `IMM8`, and returns `1` if any character in `a` was null,
/// and `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpistrs)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpistri, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpistrs<const IMM8: i32>(a: __m128i, b: __m128i) -> i32 {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { pcmpistris128(a.as_i8x16(), b.as_i8x16(), IMM8 as i8) }
}

/// Compares packed strings with implicit lengths in `a` and `b` using the
/// control in `IMM8`, and return bit `0` of the resulting bit mask.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpistro)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpistri, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpistro<const IMM8: i32>(a: __m128i, b: __m128i) -> i32 {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { pcmpistrio128(a.as_i8x16(), b.as_i8x16(), IMM8 as i8) }
}

/// Compares packed strings with implicit lengths in `a` and `b` using the
/// control in `IMM8`, and return `1` if `b` did not contain a null
/// character and the resulting mask was zero, and `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpistra)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpistri, IMM8 = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpistra<const IMM8: i32>(a: __m128i, b: __m128i) -> i32 {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { pcmpistria128(a.as_i8x16(), b.as_i8x16(), IMM8 as i8) }
}

/// Compares packed strings in `a` and `b` with lengths `la` and `lb`
/// using the control in `IMM8`, and return the generated mask.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpestrm)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpestrm, IMM8 = 0))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpestrm<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> __m128i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { transmute(pcmpestrm128(a.as_i8x16(), la, b.as_i8x16(), lb, IMM8 as i8)) }
}

/// Compares packed strings `a` and `b` with lengths `la` and `lb` using the
/// control in `IMM8` and return the generated index. Similar to
/// [`_mm_cmpistri`] with the exception that [`_mm_cmpistri`] implicitly
/// determines the length of `a` and `b`.
///
/// # Control modes
///
/// The control specified by `IMM8` may be one or more of the following.
///
/// ## Data size and signedness
///
///  - [`_SIDD_UBYTE_OPS`] - Default
///  - [`_SIDD_UWORD_OPS`]
///  - [`_SIDD_SBYTE_OPS`]
///  - [`_SIDD_SWORD_OPS`]
///
/// ## Comparison options
///  - [`_SIDD_CMP_EQUAL_ANY`] - Default
///  - [`_SIDD_CMP_RANGES`]
///  - [`_SIDD_CMP_EQUAL_EACH`]
///  - [`_SIDD_CMP_EQUAL_ORDERED`]
///
/// ## Result polarity
///  - [`_SIDD_POSITIVE_POLARITY`] - Default
///  - [`_SIDD_NEGATIVE_POLARITY`]
///
/// ## Bit returned
///  - [`_SIDD_LEAST_SIGNIFICANT`] - Default
///  - [`_SIDD_MOST_SIGNIFICANT`]
///
/// # Examples
///
/// ```
/// #[cfg(target_arch = "x86")]
/// use std::arch::x86::*;
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
///
/// # fn main() {
/// #     if is_x86_feature_detected!("sse4.2") {
/// #         #[target_feature(enable = "sse4.2")]
/// #         unsafe fn worker() {
///
/// // The string we want to find a substring in
/// let haystack = b"Split \r\n\t line  ";
///
/// // The string we want to search for with some
/// // extra bytes we do not want to search for.
/// let needle = b"\r\n\t ignore this ";
///
/// let a = _mm_loadu_si128(needle.as_ptr() as *const _);
/// let b = _mm_loadu_si128(haystack.as_ptr() as *const _);
///
/// // Note: We explicitly specify we only want to search `b` for the
/// // first 3 characters of a.
/// let idx = _mm_cmpestri(a, 3, b, 15, _SIDD_CMP_EQUAL_ORDERED);
///
/// assert_eq!(idx, 6);
/// #         }
/// #         unsafe { worker(); }
/// #     }
/// # }
/// ```
///
/// [`_SIDD_UBYTE_OPS`]: constant._SIDD_UBYTE_OPS.html
/// [`_SIDD_UWORD_OPS`]: constant._SIDD_UWORD_OPS.html
/// [`_SIDD_SBYTE_OPS`]: constant._SIDD_SBYTE_OPS.html
/// [`_SIDD_SWORD_OPS`]: constant._SIDD_SWORD_OPS.html
/// [`_SIDD_CMP_EQUAL_ANY`]: constant._SIDD_CMP_EQUAL_ANY.html
/// [`_SIDD_CMP_RANGES`]: constant._SIDD_CMP_RANGES.html
/// [`_SIDD_CMP_EQUAL_EACH`]: constant._SIDD_CMP_EQUAL_EACH.html
/// [`_SIDD_CMP_EQUAL_ORDERED`]: constant._SIDD_CMP_EQUAL_ORDERED.html
/// [`_SIDD_POSITIVE_POLARITY`]: constant._SIDD_POSITIVE_POLARITY.html
/// [`_SIDD_NEGATIVE_POLARITY`]: constant._SIDD_NEGATIVE_POLARITY.html
/// [`_SIDD_LEAST_SIGNIFICANT`]: constant._SIDD_LEAST_SIGNIFICANT.html
/// [`_SIDD_MOST_SIGNIFICANT`]: constant._SIDD_MOST_SIGNIFICANT.html
/// [`_mm_cmpistri`]: fn._mm_cmpistri.html
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpestri)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpestri, IMM8 = 0))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpestri<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32 {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { pcmpestri128(a.as_i8x16(), la, b.as_i8x16(), lb, IMM8 as i8) }
}

/// Compares packed strings in `a` and `b` with lengths `la` and `lb`
/// using the control in `IMM8`, and return `1` if any character in
/// `b` was null, and `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpestrz)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpestri, IMM8 = 0))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpestrz<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32 {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { pcmpestriz128(a.as_i8x16(), la, b.as_i8x16(), lb, IMM8 as i8) }
}

/// Compares packed strings in `a` and `b` with lengths `la` and `lb`
/// using the control in `IMM8`, and return `1` if the resulting mask
/// was non-zero, and `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpestrc)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpestri, IMM8 = 0))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpestrc<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32 {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { pcmpestric128(a.as_i8x16(), la, b.as_i8x16(), lb, IMM8 as i8) }
}

/// Compares packed strings in `a` and `b` with lengths `la` and `lb`
/// using the control in `IMM8`, and return `1` if any character in
/// a was null, and `0` otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpestrs)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpestri, IMM8 = 0))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpestrs<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32 {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { pcmpestris128(a.as_i8x16(), la, b.as_i8x16(), lb, IMM8 as i8) }
}

/// Compares packed strings in `a` and `b` with lengths `la` and `lb`
/// using the control in `IMM8`, and return bit `0` of the resulting
/// bit mask.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpestro)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpestri, IMM8 = 0))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpestro<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32 {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { pcmpestrio128(a.as_i8x16(), la, b.as_i8x16(), lb, IMM8 as i8) }
}

/// Compares packed strings in `a` and `b` with lengths `la` and `lb`
/// using the control in `IMM8`, and return `1` if `b` did not
/// contain a null character and the resulting mask was zero, and `0`
/// otherwise.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpestra)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpestri, IMM8 = 0))]
#[rustc_legacy_const_generics(4)]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpestra<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32 {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { pcmpestria128(a.as_i8x16(), la, b.as_i8x16(), lb, IMM8 as i8) }
}

/// Starting with the initial value in `crc`, return the accumulated
/// CRC32-C value for unsigned 8-bit integer `v`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_crc32_u8)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(crc32))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_crc32_u8(crc: u32, v: u8) -> u32 {
    unsafe { crc32_32_8(crc, v) }
}

/// Starting with the initial value in `crc`, return the accumulated
/// CRC32-C value for unsigned 16-bit integer `v`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_crc32_u16)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(crc32))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_crc32_u16(crc: u32, v: u16) -> u32 {
    unsafe { crc32_32_16(crc, v) }
}

/// Starting with the initial value in `crc`, return the accumulated
/// CRC32-C value for unsigned 32-bit integer `v`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_crc32_u32)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(crc32))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_crc32_u32(crc: u32, v: u32) -> u32 {
    unsafe { crc32_32_32(crc, v) }
}

/// Compares packed 64-bit integers in `a` and `b` for greater-than,
/// return the results.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_cmpgt_epi64)
#[inline]
#[target_feature(enable = "sse4.2")]
#[cfg_attr(test, assert_instr(pcmpgtq))]
#[stable(feature = "simd_x86", since = "1.27.0")]
pub fn _mm_cmpgt_epi64(a: __m128i, b: __m128i) -> __m128i {
    unsafe { transmute(simd_gt::<_, i64x2>(a.as_i64x2(), b.as_i64x2())) }
}

#[allow(improper_ctypes)]
unsafe extern "C" {
    // SSE 4.2 string and text comparison ops
    #[link_name = "llvm.x86.sse42.pcmpestrm128"]
    fn pcmpestrm128(a: i8x16, la: i32, b: i8x16, lb: i32, imm8: i8) -> u8x16;
    #[link_name = "llvm.x86.sse42.pcmpestri128"]
    fn pcmpestri128(a: i8x16, la: i32, b: i8x16, lb: i32, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpestriz128"]
    fn pcmpestriz128(a: i8x16, la: i32, b: i8x16, lb: i32, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpestric128"]
    fn pcmpestric128(a: i8x16, la: i32, b: i8x16, lb: i32, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpestris128"]
    fn pcmpestris128(a: i8x16, la: i32, b: i8x16, lb: i32, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpestrio128"]
    fn pcmpestrio128(a: i8x16, la: i32, b: i8x16, lb: i32, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpestria128"]
    fn pcmpestria128(a: i8x16, la: i32, b: i8x16, lb: i32, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistrm128"]
    fn pcmpistrm128(a: i8x16, b: i8x16, imm8: i8) -> i8x16;
    #[link_name = "llvm.x86.sse42.pcmpistri128"]
    fn pcmpistri128(a: i8x16, b: i8x16, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistriz128"]
    fn pcmpistriz128(a: i8x16, b: i8x16, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistric128"]
    fn pcmpistric128(a: i8x16, b: i8x16, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistris128"]
    fn pcmpistris128(a: i8x16, b: i8x16, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistrio128"]
    fn pcmpistrio128(a: i8x16, b: i8x16, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistria128"]
    fn pcmpistria128(a: i8x16, b: i8x16, imm8: i8) -> i32;
    // SSE 4.2 CRC instructions
    #[link_name = "llvm.x86.sse42.crc32.32.8"]
    fn crc32_32_8(crc: u32, v: u8) -> u32;
    #[link_name = "llvm.x86.sse42.crc32.32.16"]
    fn crc32_32_16(crc: u32, v: u16) -> u32;
    #[link_name = "llvm.x86.sse42.crc32.32.32"]
    fn crc32_32_32(crc: u32, v: u32) -> u32;
}

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;
    use std::ptr;

    // Currently one cannot `load` a &[u8] that is less than 16
    // in length. This makes loading strings less than 16 in length
    // a bit difficult. Rather than `load` and mutate the __m128i,
    // it is easier to memcpy the given string to a local slice with
    // length 16 and `load` the local slice.
    #[target_feature(enable = "sse4.2")]
    unsafe fn str_to_m128i(s: &[u8]) -> __m128i {
        assert!(s.len() <= 16);
        let slice = &mut [0u8; 16];
        ptr::copy_nonoverlapping(s.as_ptr(), slice.as_mut_ptr(), s.len());
        _mm_loadu_si128(slice.as_ptr() as *const _)
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_cmpistrm() {
        let a = str_to_m128i(b"Hello! Good-Bye!");
        let b = str_to_m128i(b"hello! good-bye!");
        let i = _mm_cmpistrm::<_SIDD_UNIT_MASK>(a, b);
        #[rustfmt::skip]
        let res = _mm_setr_epi8(
            0x00, !0, !0, !0, !0, !0, !0, 0x00,
            !0, !0, !0, !0, 0x00, !0, !0, !0,
        );
        assert_eq_m128i(i, res);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_cmpistri() {
        let a = str_to_m128i(b"Hello");
        let b = str_to_m128i(b"   Hello        ");
        let i = _mm_cmpistri::<_SIDD_CMP_EQUAL_ORDERED>(a, b);
        assert_eq!(3, i);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_cmpistrz() {
        let a = str_to_m128i(b"");
        let b = str_to_m128i(b"Hello");
        let i = _mm_cmpistrz::<_SIDD_CMP_EQUAL_ORDERED>(a, b);
        assert_eq!(1, i);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_cmpistrc() {
        let a = str_to_m128i(b"                ");
        let b = str_to_m128i(b"       !        ");
        let i = _mm_cmpistrc::<_SIDD_UNIT_MASK>(a, b);
        assert_eq!(1, i);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_cmpistrs() {
        let a = str_to_m128i(b"Hello");
        let b = str_to_m128i(b"");
        let i = _mm_cmpistrs::<_SIDD_CMP_EQUAL_ORDERED>(a, b);
        assert_eq!(1, i);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_cmpistro() {
        #[rustfmt::skip]
        let a_bytes = _mm_setr_epi8(
            0x00, 0x47, 0x00, 0x65, 0x00, 0x6c, 0x00, 0x6c,
            0x00, 0x6f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        );
        #[rustfmt::skip]
        let b_bytes = _mm_setr_epi8(
            0x00, 0x48, 0x00, 0x65, 0x00, 0x6c, 0x00, 0x6c,
            0x00, 0x6f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        );
        let a = a_bytes;
        let b = b_bytes;
        let i = _mm_cmpistro::<{ _SIDD_UWORD_OPS | _SIDD_UNIT_MASK }>(a, b);
        assert_eq!(0, i);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_cmpistra() {
        let a = str_to_m128i(b"");
        let b = str_to_m128i(b"Hello!!!!!!!!!!!");
        let i = _mm_cmpistra::<_SIDD_UNIT_MASK>(a, b);
        assert_eq!(1, i);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_cmpestrm() {
        let a = str_to_m128i(b"Hello!");
        let b = str_to_m128i(b"Hello.");
        let i = _mm_cmpestrm::<_SIDD_UNIT_MASK>(a, 5, b, 5);
        #[rustfmt::skip]
        let r = _mm_setr_epi8(
            !0, !0, !0, !0, !0, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        );
        assert_eq_m128i(i, r);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_cmpestri() {
        let a = str_to_m128i(b"bar - garbage");
        let b = str_to_m128i(b"foobar");
        let i = _mm_cmpestri::<_SIDD_CMP_EQUAL_ORDERED>(a, 3, b, 6);
        assert_eq!(3, i);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_cmpestrz() {
        let a = str_to_m128i(b"");
        let b = str_to_m128i(b"Hello");
        let i = _mm_cmpestrz::<_SIDD_CMP_EQUAL_ORDERED>(a, 16, b, 6);
        assert_eq!(1, i);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_cmpestrc() {
        let va = str_to_m128i(b"!!!!!!!!");
        let vb = str_to_m128i(b"        ");
        let i = _mm_cmpestrc::<_SIDD_UNIT_MASK>(va, 7, vb, 7);
        assert_eq!(0, i);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_cmpestrs() {
        #[rustfmt::skip]
        let a_bytes = _mm_setr_epi8(
            0x00, 0x48, 0x00, 0x65, 0x00, 0x6c, 0x00, 0x6c,
            0x00, 0x6f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        );
        let a = a_bytes;
        let b = _mm_set1_epi8(0x00);
        let i = _mm_cmpestrs::<_SIDD_UWORD_OPS>(a, 8, b, 0);
        assert_eq!(0, i);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_cmpestro() {
        let a = str_to_m128i(b"Hello");
        let b = str_to_m128i(b"World");
        let i = _mm_cmpestro::<_SIDD_UBYTE_OPS>(a, 5, b, 5);
        assert_eq!(0, i);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_cmpestra() {
        let a = str_to_m128i(b"Cannot match a");
        let b = str_to_m128i(b"Null after 14");
        let i = _mm_cmpestra::<{ _SIDD_CMP_EQUAL_EACH | _SIDD_UNIT_MASK }>(a, 14, b, 16);
        assert_eq!(1, i);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_crc32_u8() {
        let crc = 0x2aa1e72b;
        let v = 0x2a;
        let i = _mm_crc32_u8(crc, v);
        assert_eq!(i, 0xf24122e4);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_crc32_u16() {
        let crc = 0x8ecec3b5;
        let v = 0x22b;
        let i = _mm_crc32_u16(crc, v);
        assert_eq!(i, 0x13bb2fb);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_crc32_u32() {
        let crc = 0xae2912c8;
        let v = 0x845fed;
        let i = _mm_crc32_u32(crc, v);
        assert_eq!(i, 0xffae2ed1);
    }

    #[simd_test(enable = "sse4.2")]
    unsafe fn test_mm_cmpgt_epi64() {
        let a = _mm_setr_epi64x(0, 0x2a);
        let b = _mm_set1_epi64x(0x00);
        let i = _mm_cmpgt_epi64(a, b);
        assert_eq_m128i(i, _mm_setr_epi64x(0x00, 0xffffffffffffffffu64 as i64));
    }
}
