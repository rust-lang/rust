use std::os::raw::c_void;

use simd::*;
use v128::*;
use v64::*;

/// Provide a hint to the processor that the code sequence is a spin-wait loop.
///
/// This can help improve the performance and power consumption of spin-wait
/// loops.
#[inline]
pub unsafe fn _mm_pause() {
    pause()
}

/// Invalidate and flush the cache line that contains `p` from all levels of
/// the cache hierarchy.
#[inline]
pub unsafe fn _mm_clflush(p: *mut c_void) {
    clflush(p)
}

/// Perform a serializing operation on all load-from-memory instructions
/// that were issued prior to this instruction.
///
/// Guarantees that every load instruction that precedes, in program order, is
/// globally visible before any load instruction which follows the fence in
/// program order.
#[inline]
pub unsafe fn _mm_lfence() {
    lfence()
}

/// Perform a serializing operation on all load-from-memory and store-to-memory
/// instructions that were issued prior to this instruction.
///
/// Guarantees that every memory access that precedes, in program order, the
/// memory fence instruction is globally visible before any memory instruction
/// which follows the fence in program order.
#[inline]
pub unsafe fn _mm_mfence() {
    mfence()
}

/// Add packed 8-bit integers in "a" and "b", and return the results.
#[inline]
pub unsafe fn _mm_add_epi8(a: __m128i, b: __m128i) -> __m128i {
    simd_add(u8x16::from(a), u8x16::from(b)).as_m128i()
}

/// Add packed 16-bit integers in "a" and "b", and return the results.
#[inline]
pub unsafe fn _mm_add_epi16(a: __m128i, b: __m128i) -> __m128i {
    simd_add(u16x8::from(a), u16x8::from(b)).as_m128i()
}

/// Add packed 32-bit integers in "a" and "b", and return the results.
#[inline]
pub unsafe fn _mm_add_epi32(a: __m128i, b: __m128i) -> __m128i {
    simd_add(u32x4::from(a), u32x4::from(b)).as_m128i()
}

/// Add 64-bit integers "a" and "b", and return the results.
#[inline]
unsafe fn _mm_add_si64(_a: __m64, _b: __m64) -> __m64 {
    unimplemented!()
}

/// Add packed 64-bit integers in "a" and "b", and return the results.
#[inline]
pub unsafe fn _mm_add_epi64(a: __m128i, b: __m128i) -> __m128i {
    simd_add(u64x2::from(a), u64x2::from(b)).as_m128i()
}

/// Add packed 8-bit integers in "a" and "b" using saturation, and return the
/// results.
#[inline]
pub unsafe fn _mm_adds_epi8(a: __m128i, b: __m128i) -> __m128i {
    paddsb(i8x16::from(a), i8x16::from(b)).as_m128i()
}

/// Add packed 16-bit integers in "a" and "b" using saturation, and return the
/// results.
#[inline]
pub unsafe fn _mm_adds_epi16(a: __m128i, b: __m128i) -> __m128i {
    paddsw(i16x8::from(a), i16x8::from(b)).as_m128i()
}

/// Add packed unsigned 8-bit integers in "a" and "b" using saturation, and
/// return  the results.
#[inline]
pub unsafe fn _mm_adds_epu8(a: __m128i, b: __m128i) -> __m128i {
    paddsub(u8x16::from(a), u8x16::from(b)).as_m128i()
}

/// Add packed unsigned 16-bit integers in "a" and "b" using saturation, and
/// return the results.
#[inline]
pub unsafe fn _mm_adds_epu16(a: __m128i, b: __m128i) -> __m128i {
    paddsuw(u16x8::from(a), u16x8::from(b)).as_m128i()
}

/// Average packed unsigned 8-bit integers in "a" and "b", and return the
/// results.
#[inline]
pub unsafe fn _mm_avg_epu8(a: __m128i, b: __m128i) -> __m128i {
    pavgb(u8x16::from(a), u8x16::from(b)).as_m128i()
}

/// Average packed unsigned 16-bit integers in "a" and "b", and return the
/// results.
#[inline]
pub unsafe fn _mm_avg_epu16(a: __m128i, b: __m128i) -> __m128i {
    pavgw(u16x8::from(a), u16x8::from(b)).as_m128i()
}

/// Multiply packed signed 16-bit integers in "a" and "b", producing
/// intermediate signed 32-bit integers.
///
/// Horizontally add adjacent pairs of intermediate 32-bit integers, and pack
/// the results in "dst".
#[inline]
pub unsafe fn _mm_madd_epi16(a: __m128i, b: __m128i) -> __m128i {
    pmaddwd(i16x8::from(a), i16x8::from(b)).as_m128i()
}

/// Compare packed 16-bit integers in `a` and `b`, and return the packed
/// maximum values.
#[inline]
pub unsafe fn _mm_max_epi16(a: __m128i, b: __m128i) -> __m128i {
    pmaxsw(i16x8::from(a), i16x8::from(b)).as_m128i()
}

/// Compare packed unsigned 8-bit integers in `a` and `b`, and return the
/// packed maximum values.
#[inline]
pub unsafe fn _mm_max_epu8(a: __m128i, b: __m128i) -> __m128i {
    pmaxub(u8x16::from(a), u8x16::from(b)).as_m128i()
}

/// Compare packed 16-bit integers in `a` and `b`, and return the packed
/// minimum values.
#[inline]
pub unsafe fn _mm_min_epi16(a: __m128i, b: __m128i) -> __m128i {
    pminsw(i16x8::from(a), i16x8::from(b)).as_m128i()
}

/// Compare packed unsigned 8-bit integers in `a` and `b`, and return the
/// packed minimum values.
#[inline]
pub unsafe fn _mm_min_epu8(a: __m128i, b: __m128i) -> __m128i {
    pminub(u8x16::from(a), u8x16::from(b)).as_m128i()
}

/// Multiply the packed 16-bit integers in `a` and `b`.
///
/// The multiplication produces intermediate 32-bit integers, and returns the
/// high 16 bits of the intermediate integers.
#[inline]
pub unsafe fn _mm_mulhi_epi16(a: __m128i, b: __m128i) -> __m128i {
    pmulhw(i16x8::from(a), i16x8::from(b)).as_m128i()
}

/// Multiply the packed unsigned 16-bit integers in `a` and `b`.
///
/// The multiplication produces intermediate 32-bit integers, and returns the
/// high 16 bits of the intermediate integers.
#[inline]
pub unsafe fn _mm_mulhi_epu16(a: __m128i, b: __m128i) -> __m128i {
    pmulhuw(u16x8::from(a), u16x8::from(b)).as_m128i()
}

/// Multiply the packed 16-bit integers in `a` and `b`.
///
/// The multiplication produces intermediate 32-bit integers, and returns the
/// low 16 bits of the intermediate integers.
#[inline]
pub unsafe fn _mm_mullo_epi16(a: __m128i, b: __m128i) -> __m128i {
    simd_mul(i16x8::from(a), i16x8::from(b)).as_m128i()
}

/// Multiply the low unsigned 32-bit integers from `a` and `b`.
///
/// Return the unsigned 64-bit result.
#[inline]
unsafe fn _mm_mul_su32(_a: __m64, _b: __m64) -> __m64 {
    unimplemented!()
}

/// Multiply the low unsigned 32-bit integers from each packed 64-bit element
/// in `a` and `b`.
///
/// Return the unsigned 64-bit results.
#[inline]
pub unsafe fn _mm_mul_epu32(a: __m128i, b: __m128i) -> __m128i {
    pmuludq(u32x4::from(a), u32x4::from(b)).as_m128i()
}

/// Sum the absolute differences of packed unsigned 8-bit integers.
///
/// Compute the absolute differences of packed unsigned 8-bit integers in `a`
/// and `b`, then horizontally sum each consecutive 8 differences to produce
/// two unsigned 16-bit integers, and pack these unsigned 16-bit integers in
/// the low 16 bits of 64-bit elements returned.
#[inline]
pub unsafe fn _mm_sad_epu8(a: __m128i, b: __m128i) -> __m128i {
    psadbw(u8x16::from(a), u8x16::from(b)).as_m128i()
}














#[inline]
pub unsafe fn _mm_add_sd(a: __m128d, b: __m128d) -> __m128d {
    let (a, b) = (f64x2::from(a), f64x2::from(b));
    a.insert(0, a.extract(0) + b.extract(0)).as_m128d()
}

#[inline]
pub unsafe fn _mm_add_pd(a: __m128d, b: __m128d) -> __m128d {
    simd_add(f64x2::from(a), f64x2::from(b)).as_m128d()
}

#[inline]
pub unsafe fn _mm_load_pd(mem_addr: *const f64) -> __m128d {
    *(mem_addr as *const __m128d)
}

#[inline]
pub unsafe fn _mm_store_pd(mem_addr: *mut f64, a: __m128d) {
    *(mem_addr as *mut __m128d) = a;
}

#[allow(improper_ctypes)]
extern {
    #[link_name = "llvm.x86.sse2.pause"]
    pub fn pause();
    #[link_name = "llvm.x86.sse2.clflush"]
    pub fn clflush(p: *mut c_void);
    #[link_name = "llvm.x86.sse2.lfence"]
    pub fn lfence();
    #[link_name = "llvm.x86.sse2.mfence"]
    pub fn mfence();
    #[link_name = "llvm.x86.sse2.padds.b"]
    pub fn paddsb(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.x86.sse2.padds.w"]
    pub fn paddsw(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.x86.sse2.paddus.b"]
    pub fn paddsub(a: u8x16, b: u8x16) -> u8x16;
    #[link_name = "llvm.x86.sse2.paddus.w"]
    pub fn paddsuw(a: u16x8, b: u16x8) -> u16x8;
    #[link_name = "llvm.x86.sse2.pavg.b"]
    pub fn pavgb(a: u8x16, b: u8x16) -> u8x16;
    #[link_name = "llvm.x86.sse2.pavg.w"]
    pub fn pavgw(a: u16x8, b: u16x8) -> u16x8;
    #[link_name = "llvm.x86.sse2.pmadd.wd"]
    pub fn pmaddwd(a: i16x8, b: i16x8) -> i32x4;
    #[link_name = "llvm.x86.sse2.pmaxs.w"]
    pub fn pmaxsw(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.x86.sse2.pmaxu.b"]
    pub fn pmaxub(a: u8x16, b: u8x16) -> u8x16;
    #[link_name = "llvm.x86.sse2.pmins.w"]
    pub fn pminsw(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.x86.sse2.pminu.b"]
    pub fn pminub(a: u8x16, b: u8x16) -> u8x16;
    #[link_name = "llvm.x86.sse2.pmulh.w"]
    pub fn pmulhw(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.x86.sse2.pmulhu.w"]
    pub fn pmulhuw(a: u16x8, b: u16x8) -> u16x8;
    #[link_name = "llvm.x86.sse2.pmulu.dq"]
    pub fn pmuludq(a: u32x4, b: u32x4) -> u64x2;
    #[link_name = "llvm.x86.sse2.psad.bw"]
    pub fn psadbw(a: u8x16, b: u8x16) -> u64x2;
}

#[cfg(test)]
mod tests {
    use std::os::raw::c_void;

    use v128::*;
    use v64::*;
    use x86::sse2 as sse2;

    #[test]
    fn _mm_pause() {
        unsafe { sse2::_mm_pause() }
    }

    #[test]
    fn _mm_clflush() {
        let x = 0;
        unsafe { sse2::_mm_clflush(&x as *const _ as *mut c_void) }
    }

    #[test]
    fn _mm_lfence() {
        unsafe { sse2::_mm_lfence() }
    }

    #[test]
    fn _mm_mfence() {
        unsafe { sse2::_mm_mfence() }
    }

    #[test]
    fn _mm_add_epi8() {
        let a = u8x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = u8x16::new(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = unsafe { sse2::_mm_add_epi8(a.as_m128i(), b.as_m128i()) };
        let e = u8x16::new(
            16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46);
        assert_eq!(u8x16::from(r), e);
    }

    #[test]
    fn _mm_adds_epi8_overflow() {
        let a = u8x16::splat(0xFF);
        let b = u8x16::splat(1);
        let r = unsafe { sse2::_mm_adds_epi8(a.as_m128i(), b.as_m128i()) };
        assert_eq!(u8x16::from(r), u8x16::splat(0));
    }

    #[test]
    fn _mm_add_epi16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = u16x8::new(8, 9, 10, 11, 12, 13, 14, 15);
        let r = unsafe { sse2::_mm_add_epi16(a.as_m128i(), b.as_m128i()) };
        let e = u16x8::new(8, 10, 12, 14, 16, 18, 20, 22);
        assert_eq!(u16x8::from(r), e);
    }

    #[test]
    fn _mm_add_epi32() {
        let a = u32x4::new(0, 1, 2, 3);
        let b = u32x4::new(4, 5, 6, 7);
        let r = unsafe { sse2::_mm_add_epi32(a.as_m128i(), b.as_m128i()) };
        let e = u32x4::new(4, 6, 8, 10);
        assert_eq!(u32x4::from(r), e);
    }

    #[test]
    #[ignore]
    fn _mm_add_si64() {
        let (a, b) = (u64x1::new(1), u64x1::new(2));
        let r = unsafe { sse2::_mm_add_si64(a.as_m64(), b.as_m64()) };
        let e = u64x1::new(3);
        assert_eq!(u64x1::from(r), e);
    }

    #[test]
    fn _mm_add_epi64() {
        let a = u64x2::new(0, 1);
        let b = u64x2::new(2, 3);
        let r = unsafe { sse2::_mm_add_epi64(a.as_m128i(), b.as_m128i()) };
        let e = u64x2::new(2, 4);
        assert_eq!(u64x2::from(r), e);
    }

    #[test]
    fn _mm_adds_epi8() {
        let a = i8x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = i8x16::new(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = unsafe { sse2::_mm_adds_epi8(a.as_m128i(), b.as_m128i()) };
        let e = i8x16::new(
            16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46);
        assert_eq!(i8x16::from(r), e);
    }

    #[test]
    fn _mm_adds_epi8_saturate_positive() {
        let a = i8x16::splat(0x7F);
        let b = i8x16::splat(1);
        let r = unsafe { sse2::_mm_adds_epi8(a.as_m128i(), b.as_m128i()) };
        assert_eq!(i8x16::from(r), a);
    }

    #[test]
    fn _mm_adds_epi8_saturate_negative() {
        let a = i8x16::splat(-0x80);
        let b = i8x16::splat(-1);
        let r = unsafe { sse2::_mm_adds_epi8(a.as_m128i(), b.as_m128i()) };
        assert_eq!(i8x16::from(r), a);
    }

    #[test]
    fn _mm_adds_epi16() {
        let a = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = i16x8::new(8, 9, 10, 11, 12, 13, 14, 15);
        let r = unsafe { sse2::_mm_adds_epi16(a.as_m128i(), b.as_m128i()) };
        let e = i16x8::new(8, 10, 12, 14, 16, 18, 20, 22);
        assert_eq!(i16x8::from(r), e);
    }

    #[test]
    fn _mm_adds_epi16_saturate_positive() {
        let a = i16x8::splat(0x7FFF);
        let b = i16x8::splat(1);
        let r = unsafe { sse2::_mm_adds_epi16(a.as_m128i(), b.as_m128i()) };
        assert_eq!(i16x8::from(r), a);
    }

    #[test]
    fn _mm_adds_epi16_saturate_negative() {
        let a = i16x8::splat(-0x8000);
        let b = i16x8::splat(-1);
        let r = unsafe { sse2::_mm_adds_epi16(a.as_m128i(), b.as_m128i()) };
        assert_eq!(i16x8::from(r), a);
    }

    #[test]
    fn _mm_adds_epu8() {
        let a = u8x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = u8x16::new(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = unsafe { sse2::_mm_adds_epu8(a.as_m128i(), b.as_m128i()) };
        let e = u8x16::new(
            16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46);
        assert_eq!(u8x16::from(r), e);
    }

    #[test]
    fn _mm_adds_epu8_saturate() {
        let a = u8x16::splat(0xFF);
        let b = u8x16::splat(1);
        let r = unsafe { sse2::_mm_adds_epu8(a.as_m128i(), b.as_m128i()) };
        assert_eq!(u8x16::from(r), a);
    }

    #[test]
    fn _mm_adds_epu16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = u16x8::new(8, 9, 10, 11, 12, 13, 14, 15);
        let r = unsafe { sse2::_mm_adds_epu16(a.as_m128i(), b.as_m128i()) };
        let e = u16x8::new(8, 10, 12, 14, 16, 18, 20, 22);
        assert_eq!(u16x8::from(r), e);
    }

    #[test]
    fn _mm_adds_epu16_saturate() {
        let a = u16x8::splat(0xFFFF);
        let b = u16x8::splat(1);
        let r = unsafe { sse2::_mm_adds_epu16(a.as_m128i(), b.as_m128i()) };
        assert_eq!(u16x8::from(r), a);
    }

    #[test]
    fn _mm_avg_epu8() {
        let (a, b) = (u8x16::splat(3), u8x16::splat(9));
        let r = unsafe { sse2::_mm_avg_epu8(a.as_m128i(), b.as_m128i()) };
        assert_eq!(u8x16::from(r), u8x16::splat(6));
    }

    #[test]
    fn _mm_avg_epu16() {
        let (a, b) = (u16x8::splat(3), u16x8::splat(9));
        let r = unsafe { sse2::_mm_avg_epu8(a.as_m128i(), b.as_m128i()) };
        assert_eq!(u16x8::from(r), u16x8::splat(6));
    }

    #[test]
    fn _mm_madd_epi16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { sse2::_mm_madd_epi16(a.as_m128i(), b.as_m128i()) };
        let e = i32x4::new(29, 81, 149, 233);
        assert_eq!(i32x4::from(r), e);
    }

    #[test]
    fn _mm_max_epi16() {
        let a = i16x8::splat(1);
        let b = i16x8::splat(-1);
        let r = unsafe { sse2::_mm_max_epi16(a.as_m128i(), b.as_m128i()) };
        assert_eq!(i16x8::from(r), a);
    }

    #[test]
    fn _mm_max_epu8() {
        let a = u8x16::splat(1);
        let b = u8x16::splat(255);
        let r = unsafe { sse2::_mm_max_epu8(a.as_m128i(), b.as_m128i()) };
        assert_eq!(u8x16::from(r), b);
    }

    #[test]
    fn _mm_min_epi16() {
        let a = i16x8::splat(1);
        let b = i16x8::splat(-1);
        let r = unsafe { sse2::_mm_min_epi16(a.as_m128i(), b.as_m128i()) };
        assert_eq!(i16x8::from(r), b);
    }

    #[test]
    fn _mm_min_epu8() {
        let a = u8x16::splat(1);
        let b = u8x16::splat(255);
        let r = unsafe { sse2::_mm_min_epu8(a.as_m128i(), b.as_m128i()) };
        assert_eq!(u8x16::from(r), a);
    }

    #[test]
    fn _mm_mulhi_epi16() {
        let (a, b) = (i16x8::splat(1000), i16x8::splat(-1001));
        let r = unsafe { sse2::_mm_mulhi_epi16(a.as_m128i(), b.as_m128i()) };
        assert_eq!(i16x8::from(r), i16x8::splat(-16));
    }

    #[test]
    fn _mm_mulhi_epu16() {
        let (a, b) = (u16x8::splat(1000), u16x8::splat(1001));
        let r = unsafe { sse2::_mm_mulhi_epu16(a.as_m128i(), b.as_m128i()) };
        assert_eq!(u16x8::from(r), u16x8::splat(15));
    }

    #[test]
    fn _mm_mullo_epi16() {
        let (a, b) = (i16x8::splat(1000), i16x8::splat(-1001));
        let r = unsafe { sse2::_mm_mullo_epi16(a.as_m128i(), b.as_m128i()) };
        assert_eq!(i16x8::from(r), i16x8::splat(-17960));
    }

    #[test]
    #[ignore]
    fn _mm_mul_su32() {
        let a = u32x2::new(1_000_000_000, 3);
        let b = u32x2::new(1_000_000_000, 4);
        let r = unsafe { sse2::_mm_mul_su32(a.as_m64(), b.as_m64()) };
        let e = u64x1::new(1_000_000_000 * 1_000_000_000);
        assert_eq!(u64x1::from(r), e);
    }

    #[test]
    fn _mm_mul_epu32() {
        let a = u64x2::new(1_000_000_000, 1 << 34);
        let b = u64x2::new(1_000_000_000, 1 << 35);
        let r = unsafe { sse2::_mm_mul_epu32(a.as_m128i(), b.as_m128i()) };
        let e = u64x2::new(1_000_000_000 * 1_000_000_000, 0);
        assert_eq!(u64x2::from(r), e);
    }

    #[test]
    fn _mm_sad_epu8() {
        let a = u8x16::new(
            255, 254, 253, 252, 1, 2, 3, 4,
            155, 154, 153, 152, 1, 2, 3, 4);
        let b = u8x16::new(
            0, 0, 0, 0, 2, 1, 2, 1,
            1, 1, 1, 1, 1, 2, 1, 2);
        let r = unsafe { sse2::_mm_sad_epu8(a.as_m128i(), b.as_m128i()) };
        let e = u64x2::new(1020, 614);
        assert_eq!(u64x2::from(r), e);
    }
}
