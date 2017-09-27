#[cfg(test)]
use stdsimd_test::assert_instr;

use x86::__m128i;

pub const _SIDD_UBYTE_OPS: i8 = 0b00000000;
pub const _SIDD_UWORD_OPS: i8 = 0b00000001;
pub const _SIDD_SBYTE_OPS: i8 = 0b00000010;
pub const _SIDD_SWORD_OPS: i8 = 0b00000011;

pub const _SIDD_CMP_EQUAL_ANY: i8 = 0b00000000;
pub const _SIDD_CMP_RANGES: i8 = 0b00000100;
pub const _SIDD_CMP_EQUAL_EACH: i8 = 0b00001000;
pub const _SIDD_CMP_EQUAL_ORDERED: i8 = 0b00001100;

pub const _SIDD_POSITIVE_POLARITY: i8 = 0b00000000;
pub const _SIDD_NEGATIVE_POLARITY: i8 = 0b00010000;
pub const _SIDD_MASKED_NEGATIVE_POLARITY: i8 = 0b00110000;

pub const _SIDD_LEAST_SIGNIFICANT: i8 = 0b00000000;
pub const _SIDD_MOST_SIGNIFICANT: i8 = 0b01000000;

#[inline(always)]
#[target_feature = "+sse4.2"]
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

#[cfg(test)]
#[target_feature = "+sse4.2"]
#[cfg_attr(test, assert_instr(pcmpestri))]
fn _test_mm_cmpestri(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32 {
    unsafe { _mm_cmpestri(a, la, b, lb, 0) }
}

#[allow(improper_ctypes)]
extern {
    #[link_name = "llvm.x86.sse42.pcmpestri128"]
    fn pcmpestri128(a: __m128i, la: i32, b: __m128i, lb: i32, imm8: i8) -> i32;
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use v128::*;
    use x86::{__m128i, sse42};

    #[simd_test = "sse4.2"]
    fn _mm_cmpestri() {
        let a = &b"bar             "[..];
        let b = &b"foobar          "[..];
        let va = __m128i::from(u8x16::load(a, 0));
        let vb = __m128i::from(u8x16::load(b, 0));
        let i = unsafe {
            sse42::_mm_cmpestri(
                va, 3, vb, 6, sse42::_SIDD_CMP_EQUAL_ORDERED)
        };
        assert_eq!(3, i);
    }
}
