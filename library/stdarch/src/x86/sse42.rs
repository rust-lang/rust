#[cfg(test)]
use stdsimd_test::assert_instr;

use v128::*;
use x86::__m128i;

/// String contains unsigned 8-bit characters
pub const _SIDD_UBYTE_OPS: i8 = 0b00000000;
/// String contains unsigned 16-bit characters
pub const _SIDD_UWORD_OPS: i8 = 0b00000001;
/// String contains signed 8-bit characters
pub const _SIDD_SBYTE_OPS: i8 = 0b00000010;
/// String contains unsigned 16-bit characters
pub const _SIDD_SWORD_OPS: i8 = 0b00000011;

/// For each character in `a`, find if it is in `b`
pub const _SIDD_CMP_EQUAL_ANY: i8 = 0b00000000;
/// For each character in `a`, determine if `b[0] <= c <= b[1] or b[1] <= c <= b[2]...`
pub const _SIDD_CMP_RANGES: i8 = 0b00000100;
/// String equality
pub const _SIDD_CMP_EQUAL_EACH: i8 = 0b00001000;
/// Substring search
pub const _SIDD_CMP_EQUAL_ORDERED: i8 = 0b00001100;

/// Do not negate results
pub const _SIDD_POSITIVE_POLARITY: i8 = 0b00000000;
/// Negate results
pub const _SIDD_NEGATIVE_POLARITY: i8 = 0b00010000;
/// Do not negate results before the end of the string
pub const _SIDD_MASKED_POSITIVE_POLARITY: i8 = 0b00100000;
/// Negate results only before the end of the string
pub const _SIDD_MASKED_NEGATIVE_POLARITY: i8 = 0b00110000;

/// Index only: return the least significant bit
pub const _SIDD_LEAST_SIGNIFICANT: i8 = 0b00000000;
/// Index only: return the most significant bit
pub const _SIDD_MOST_SIGNIFICANT: i8 = 0b01000000;

/// Mask only: return the bit mask
pub const _SIDD_BIT_MASK: i8 = 0b00000000;
/// Mask only: return the byte mask
pub const _SIDD_UNIT_MASK: i8 = 0b01000000;

/// Compare packed strings with implicit lengths in `a` and `b` using the
/// control in `imm8`, and return the generated mask.
#[inline(always)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpistrm, imm8 = 0))]
pub unsafe fn _mm_cmpistrm(
    a: __m128i,
    b: __m128i,
    imm8: i8,
) -> u8x16 {
    macro_rules! call {
        ($imm8:expr) => { pcmpistrm128(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Compare packed strings with implicit lengths in `a` and `b` using the
/// control in `imm8`, and return the generated index.
#[inline(always)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpistri, imm8 = 0))]
pub unsafe fn _mm_cmpistri(
    a: __m128i,
    b: __m128i,
    imm8: i8,
) -> i32 {
    macro_rules! call {
        ($imm8:expr) => { pcmpistri128(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Compare packed strings with implicit lengths in `a` and `b` using the
/// control in `imm8`, and return `1` if any character in `b` was null.
/// and `0` otherwise.
#[inline(always)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpistri, imm8 = 0))]
pub unsafe fn _mm_cmpistrz(
    a: __m128i,
    b: __m128i,
    imm8: i8,
) -> i32 {
    macro_rules! call {
        ($imm8:expr) => { pcmpistriz128(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Compare packed strings with implicit lengths in `a` and `b` using the
/// control in `imm8`, and return `1` if the resulting mask was non-zero,
/// and `0` otherwise.
#[inline(always)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpistri, imm8 = 0))]
pub unsafe fn _mm_cmpistrc(
    a: __m128i,
    b: __m128i,
    imm8: i8,
) -> i32 {
    macro_rules! call {
        ($imm8:expr) => { pcmpistric128(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Compare packed strings with implicit lengths in `a` and `b` using the
/// control in `imm8`, and returns `1` if any character in `a` was null,
/// and `0` otherwise.
#[inline(always)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpistri, imm8 = 0))]
pub unsafe fn _mm_cmpistrs(
    a: __m128i,
    b: __m128i,
    imm8: i8,
) -> i32 {
    macro_rules! call {
        ($imm8:expr) => { pcmpistris128(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Compare packed strings with implicit lengths in `a` and `b` using the
/// control in `imm8`, and return bit `0` of the resulting bit mask.
#[inline(always)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpistri, imm8 = 0))]
pub unsafe fn _mm_cmpistro(
    a: __m128i,
    b: __m128i,
    imm8: i8,
) -> i32 {
    macro_rules! call {
        ($imm8:expr) => { pcmpistrio128(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Compare packed strings with implicit lengths in `a` and `b` using the
/// control in `imm8`, and return `1` if `b` did not contain a null
/// character and the resulting mask was zero, and `0` otherwise.
#[inline(always)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpistri, imm8 = 0))]
pub unsafe fn _mm_cmpistra(
    a: __m128i,
    b: __m128i,
    imm8: i8,
) -> i32 {
    macro_rules! call {
        ($imm8:expr) => { pcmpistria128(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Compare packed strings in `a` and `b` with lengths `la` and `lb`
/// using the control in `imm8`, and return the generated mask.
#[inline(always)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpestrm, imm8 = 0))]
pub unsafe fn _mm_cmpestrm(
    a: __m128i,
    la: i32,
    b: __m128i,
    lb: i32,
    imm8: i8,
) -> u8x16 {
    macro_rules! call {
        ($imm8:expr) => { pcmpestrm128(a, la, b, lb, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Compare packed strings `a` and `b` with lengths `la` and `lb` using the
/// control in `imm8`, and return the generated index.
#[inline(always)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpestri, imm8 = 0))]
pub unsafe fn _mm_cmpestri(
    a: __m128i,
    la: i32,
    b: __m128i,
    lb: i32,
    imm8: i8,
) -> i32 {
    macro_rules! call {
        ($imm8:expr) => { pcmpestri128(a, la, b, lb, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Compare packed strings in `a` and `b` with lengths `la` and `lb`
/// using the control in `imm8`, and return `1` if any character in
/// `b` was null, and `0` otherwise.
#[inline(always)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpestri, imm8 = 0))]
pub unsafe fn _mm_cmpestrz(
    a: __m128i,
    la: i32,
    b: __m128i,
    lb: i32,
    imm8: i8,
) -> i32 {
    macro_rules! call {
        ($imm8:expr) => { pcmpestriz128(a, la, b, lb, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Compare packed strings in `a` and `b` with lengths `la` and `lb`
/// using the control in `imm8`, and return `1` if the resulting mask
/// was non-zero, and `0` otherwise.
#[inline(always)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpestri, imm8 = 0))]
pub unsafe fn _mm_cmpestrc(
    a: __m128i,
    la: i32,
    b: __m128i,
    lb: i32,
    imm8: i8,
) -> i32 {
    macro_rules! call {
        ($imm8:expr) => { pcmpestric128(a, la, b, lb, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Compare packed strings in `a` and `b` with lengths `la` and `lb`
/// using the control in `imm8`, and return `1` if any character in
/// a was null, and `0` otherwise.
#[inline(always)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpestri, imm8 = 0))]
pub unsafe fn _mm_cmpestrs(
    a: __m128i,
    la: i32,
    b: __m128i,
    lb: i32,
    imm8: i8,
) -> i32 {
    macro_rules! call {
        ($imm8:expr) => { pcmpestris128(a, la, b, lb, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Compare packed strings in `a` and `b` with lengths `la` and `lb`
/// using the control in `imm8`, and return bit `0` of the resulting
/// bit mask.
#[inline(always)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpestri, imm8 = 0))]
pub unsafe fn _mm_cmpestro(
    a: __m128i,
    la: i32,
    b: __m128i,
    lb: i32,
    imm8: i8,
) -> i32 {
    macro_rules! call {
        ($imm8:expr) => { pcmpestrio128(a, la, b, lb, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Compare packed strings in `a` and `b` with lengths `la` and `lb`
/// using the control in `imm8`, and return `1` if `b` did not
/// contain a null character and the resulting mask was zero, and `0`
/// otherwise.
#[inline(always)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpestri, imm8 = 0))]
pub unsafe fn _mm_cmpestra(
    a: __m128i,
    la: i32,
    b: __m128i,
    lb: i32,
    imm8: i8,
) -> i32 {
    macro_rules! call {
        ($imm8:expr) => { pcmpestria128(a, la, b, lb, $imm8) }
    }
    constify_imm8!(imm8, call)
}

#[allow(improper_ctypes)]
extern {
    #[link_name = "llvm.x86.sse42.pcmpestrm128"]
    fn pcmpestrm128(a: __m128i, la: i32, b: __m128i, lb: i32, imm8: i8) -> u8x16;
    #[link_name = "llvm.x86.sse42.pcmpestri128"]
    fn pcmpestri128(a: __m128i, la: i32, b: __m128i, lb: i32, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpestriz128"]
    fn pcmpestriz128(a: __m128i, la: i32, b: __m128i, lb: i32, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpestric128"]
    fn pcmpestric128(a: __m128i, la: i32, b: __m128i, lb: i32, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpestris128"]
    fn pcmpestris128(a: __m128i, la: i32, b: __m128i, lb: i32, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpestrio128"]
    fn pcmpestrio128(a: __m128i, la: i32, b: __m128i, lb: i32, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpestria128"]
    fn pcmpestria128(a: __m128i, la: i32, b: __m128i, lb: i32, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistrm128"]
    fn pcmpistrm128(a: __m128i, b: __m128i, imm8: i8) -> u8x16;
    #[link_name = "llvm.x86.sse42.pcmpistri128"]
    fn pcmpistri128(a: __m128i, b: __m128i, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistriz128"]
    fn pcmpistriz128(a: __m128i, b: __m128i, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistric128"]
    fn pcmpistric128(a: __m128i, b: __m128i, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistris128"]
    fn pcmpistris128(a: __m128i, b: __m128i, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistrio128"]
    fn pcmpistrio128(a: __m128i, b: __m128i, imm8: i8) -> i32;
    #[link_name = "llvm.x86.sse42.pcmpistria128"]
    fn pcmpistria128(a: __m128i, b: __m128i, imm8: i8) -> i32;
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use std::ptr;
    use v128::*;
    use x86::{__m128i, sse42};

    // Currently one cannot `load` a &[u8] that is is less than 16
    // in length. This makes loading strings less than 16 in length
    // a bit difficult. Rather than `load` and mutate the __m128i,
    // it is easier to memcpy the given string to a local slice with
    // length 16 and `load` the local slice.
    unsafe fn str_to_m128i(s: &[u8]) -> __m128i {
        assert!(s.len() <= 16);
        let slice = &mut [0u8; 16];
        ptr::copy_nonoverlapping(
            s.get_unchecked(0) as *const u8 as *const u8,
            slice.get_unchecked_mut(0) as *mut u8 as *mut u8,
            s.len());
        __m128i::from(u8x16::load(slice, 0))
    }

    #[simd_test = "sse4.2"]
    unsafe fn _mm_cmpistrm() {
        let a = str_to_m128i(b"Hello! Good-Bye!");
        let b = str_to_m128i(b"hello! good-bye!");
        let i = sse42::_mm_cmpistrm(a, b, sse42::_SIDD_UNIT_MASK);
        let res = u8x16::new(0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00,
                             0xff, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff);
        assert_eq!(i, res);
    }

    #[simd_test = "sse4.2"]
    unsafe fn _mm_cmpistri() {
        let a = str_to_m128i(b"Hello");
        let b = str_to_m128i(b"   Hello        ");
        let i = sse42::_mm_cmpistri(a, b, sse42::_SIDD_CMP_EQUAL_ORDERED);
        assert_eq!(3, i);
    }

    #[simd_test = "sse4.2"]
    unsafe fn _mm_cmpistrz() {
        let a = str_to_m128i(b"");
        let b = str_to_m128i(b"Hello");
        let i = sse42::_mm_cmpistrz(a, b, sse42::_SIDD_CMP_EQUAL_ORDERED);
        assert_eq!(1, i);
    }

    #[simd_test = "sse4.2"]
    unsafe fn _mm_cmpistrc() {
        let a = str_to_m128i(b"                ");
        let b = str_to_m128i(b"       !        ");
        let i = sse42::_mm_cmpistrc(a, b, sse42::_SIDD_UNIT_MASK);
        assert_eq!(1, i);
    }

    #[simd_test = "sse4.2"]
    unsafe fn _mm_cmpistrs() {
        let a = str_to_m128i(b"Hello");
        let b = str_to_m128i(b"");
        let i = sse42::_mm_cmpistrs(a, b, sse42::_SIDD_CMP_EQUAL_ORDERED);
        assert_eq!(1, i);
    }

    #[simd_test = "sse4.2"]
    unsafe fn _mm_cmpistro() {
        let a_bytes = u8x16::new(0x00, 0x47, 0x00, 0x65, 0x00, 0x6c, 0x00, 0x6c,
                                 0x00, 0x6f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let b_bytes = u8x16::new(0x00, 0x48, 0x00, 0x65, 0x00, 0x6c, 0x00, 0x6c,
                                 0x00, 0x6f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let a = __m128i::from(a_bytes);
        let b = __m128i::from(b_bytes);
        let i = sse42::_mm_cmpistro(
                a, b, sse42::_SIDD_UWORD_OPS | sse42::_SIDD_UNIT_MASK);
        assert_eq!(0, i);
    }

    #[simd_test = "sse4.2"]
    unsafe fn _mm_cmpistra() {
        let a = str_to_m128i(b"");
        let b = str_to_m128i(b"Hello!!!!!!!!!!!");
        let i = sse42::_mm_cmpistra(a, b, sse42::_SIDD_UNIT_MASK);
        assert_eq!(1, i);
    }

    #[simd_test = "sse4.2"]
    unsafe fn _mm_cmpestrm() {
        let a = str_to_m128i(b"Hello!");
        let b = str_to_m128i(b"Hello.");
        let i = sse42::_mm_cmpestrm(a, 5, b, 5, sse42::_SIDD_UNIT_MASK);
        assert_eq!(i, u8x16::new(0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00,
                                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00));
    }

    #[simd_test = "sse4.2"]
    unsafe fn _mm_cmpestri() {
        let a = str_to_m128i(b"bar - garbage");
        let b = str_to_m128i(b"foobar");
        let i = sse42::_mm_cmpestri(a, 3, b, 6, sse42::_SIDD_CMP_EQUAL_ORDERED);
        assert_eq!(3, i);
    }

    #[simd_test = "sse4.2"]
    unsafe fn _mm_cmpestrz() {
        let a = str_to_m128i(b"");
        let b = str_to_m128i(b"Hello");
        let i = sse42::_mm_cmpestrz(
                a, 16, b, 6, sse42::_SIDD_CMP_EQUAL_ORDERED);
        assert_eq!(1, i);
    }

    #[simd_test = "sse4.2"]
    unsafe fn _mm_cmpestrc() {
        let va = str_to_m128i(b"!!!!!!!!");
        let vb = str_to_m128i(b"        ");
        let i = sse42::_mm_cmpestrc(
                va, 7, vb, 7, sse42::_SIDD_UNIT_MASK);
        assert_eq!(0, i);
    }

    #[simd_test = "sse4.2"]
    unsafe fn _mm_cmpestrs() {
        let a_bytes = u8x16::new(0x00, 0x48, 0x00, 0x65, 0x00, 0x6c, 0x00, 0x6c,
                                 0x00, 0x6f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let a = __m128i::from(a_bytes);
        let b = __m128i::from(u8x16::splat(0x00));
        let i = sse42::_mm_cmpestrs(
                a, 8, b, 0, sse42::_SIDD_UWORD_OPS);
        assert_eq!(0, i);
    }

    #[simd_test = "sse4.2"]
    unsafe fn _mm_cmpestro() {
        let a = str_to_m128i(b"Hello");
        let b = str_to_m128i(b"World");
        let i = sse42::_mm_cmpestro(
                a, 5, b, 5, sse42::_SIDD_UBYTE_OPS);
        assert_eq!(0, i);
    }

    #[simd_test = "sse4.2"]
    unsafe fn _mm_cmpestra() {
        let a = str_to_m128i(b"Cannot match a");
        let b = str_to_m128i(b"Null after 14");
        let i = sse42::_mm_cmpestra(
                a, 14, b, 16, sse42::_SIDD_CMP_EQUAL_EACH | sse42::_SIDD_UNIT_MASK);
        assert_eq!(1, i);
    }
}
