use std::mem;
use std::os::raw::c_void;
use std::ptr;

use simd::*;
use v128::*;
use v64::*;

/// Provide a hint to the processor that the code sequence is a spin-wait loop.
///
/// This can help improve the performance and power consumption of spin-wait
/// loops.
#[inline(always)]
pub unsafe fn _mm_pause() {
    pause()
}

/// Invalidate and flush the cache line that contains `p` from all levels of
/// the cache hierarchy.
#[inline(always)]
pub unsafe fn _mm_clflush(p: *mut c_void) {
    clflush(p)
}

/// Perform a serializing operation on all load-from-memory instructions
/// that were issued prior to this instruction.
///
/// Guarantees that every load instruction that precedes, in program order, is
/// globally visible before any load instruction which follows the fence in
/// program order.
#[inline(always)]
pub unsafe fn _mm_lfence() {
    lfence()
}

/// Perform a serializing operation on all load-from-memory and store-to-memory
/// instructions that were issued prior to this instruction.
///
/// Guarantees that every memory access that precedes, in program order, the
/// memory fence instruction is globally visible before any memory instruction
/// which follows the fence in program order.
#[inline(always)]
pub unsafe fn _mm_mfence() {
    mfence()
}

/// Add packed 8-bit integers in "a" and "b", and return the results.
#[inline(always)]
pub unsafe fn _mm_add_epi8(a: i8x16, b: i8x16) -> i8x16 {
    simd_add(a, b)
}

/// Add packed 16-bit integers in "a" and "b", and return the results.
#[inline(always)]
pub unsafe fn _mm_add_epi16(a: i16x8, b: i16x8) -> i16x8 {
    simd_add(a, b)
}

/// Add packed 32-bit integers in "a" and "b", and return the results.
#[inline(always)]
pub unsafe fn _mm_add_epi32(a: i32x4, b: i32x4) -> i32x4 {
    simd_add(a, b)
}

/// Add packed 64-bit integers in "a" and "b", and return the results.
#[inline(always)]
pub unsafe fn _mm_add_epi64(a: i64x2, b: i64x2) -> i64x2 {
    simd_add(a, b)
}

/// Add packed 8-bit integers in "a" and "b" using saturation, and return the
/// results.
#[inline(always)]
pub unsafe fn _mm_adds_epi8(a: i8x16, b: i8x16) -> i8x16 {
    paddsb(a, b)
}

/// Add packed 16-bit integers in "a" and "b" using saturation, and return the
/// results.
#[inline(always)]
pub unsafe fn _mm_adds_epi16(a: i16x8, b: i16x8) -> i16x8 {
    paddsw(a, b)
}

/// Add packed unsigned 8-bit integers in "a" and "b" using saturation, and
/// return  the results.
#[inline(always)]
pub unsafe fn _mm_adds_epu8(a: u8x16, b: u8x16) -> u8x16 {
    paddsub(a, b)
}

/// Add packed unsigned 16-bit integers in "a" and "b" using saturation, and
/// return the results.
#[inline(always)]
pub unsafe fn _mm_adds_epu16(a: u16x8, b: u16x8) -> u16x8 {
    paddsuw(a, b)
}

/// Average packed unsigned 8-bit integers in "a" and "b", and return the
/// results.
#[inline(always)]
pub unsafe fn _mm_avg_epu8(a: u8x16, b: u8x16) -> u8x16 {
    pavgb(a, b)
}

/// Average packed unsigned 16-bit integers in "a" and "b", and return the
/// results.
#[inline(always)]
pub unsafe fn _mm_avg_epu16(a: u16x8, b: u16x8) -> u16x8 {
    pavgw(a, b)
}

/// Multiply packed signed 16-bit integers in "a" and "b", producing
/// intermediate signed 32-bit integers.
///
/// Horizontally add adjacent pairs of intermediate 32-bit integers, and pack
/// the results in "dst".
#[inline(always)]
pub unsafe fn _mm_madd_epi16(a: i16x8, b: i16x8) -> i32x4 {
    pmaddwd(a, b)
}

/// Compare packed 16-bit integers in `a` and `b`, and return the packed
/// maximum values.
#[inline(always)]
pub unsafe fn _mm_max_epi16(a: i16x8, b: i16x8) -> i16x8 {
    pmaxsw(a, b)
}

/// Compare packed unsigned 8-bit integers in `a` and `b`, and return the
/// packed maximum values.
#[inline(always)]
pub unsafe fn _mm_max_epu8(a: u8x16, b: u8x16) -> u8x16 {
    pmaxub(a, b)
}

/// Compare packed 16-bit integers in `a` and `b`, and return the packed
/// minimum values.
#[inline(always)]
pub unsafe fn _mm_min_epi16(a: i16x8, b: i16x8) -> i16x8 {
    pminsw(a, b)
}

/// Compare packed unsigned 8-bit integers in `a` and `b`, and return the
/// packed minimum values.
#[inline(always)]
pub unsafe fn _mm_min_epu8(a: u8x16, b: u8x16) -> u8x16 {
    pminub(a, b)
}

/// Multiply the packed 16-bit integers in `a` and `b`.
///
/// The multiplication produces intermediate 32-bit integers, and returns the
/// high 16 bits of the intermediate integers.
#[inline(always)]
pub unsafe fn _mm_mulhi_epi16(a: i16x8, b: i16x8) -> i16x8 {
    pmulhw(a, b)
}

/// Multiply the packed unsigned 16-bit integers in `a` and `b`.
///
/// The multiplication produces intermediate 32-bit integers, and returns the
/// high 16 bits of the intermediate integers.
#[inline(always)]
pub unsafe fn _mm_mulhi_epu16(a: u16x8, b: u16x8) -> u16x8 {
    pmulhuw(a, b)
}

/// Multiply the packed 16-bit integers in `a` and `b`.
///
/// The multiplication produces intermediate 32-bit integers, and returns the
/// low 16 bits of the intermediate integers.
#[inline(always)]
pub unsafe fn _mm_mullo_epi16(a: i16x8, b: i16x8) -> i16x8 {
    simd_mul(a, b)
}

/// Multiply the low unsigned 32-bit integers from each packed 64-bit element
/// in `a` and `b`.
///
/// Return the unsigned 64-bit results.
#[inline(always)]
pub unsafe fn _mm_mul_epu32(a: u32x4, b: u32x4) -> u64x2 {
    pmuludq(a, b)
}

/// Sum the absolute differences of packed unsigned 8-bit integers.
///
/// Compute the absolute differences of packed unsigned 8-bit integers in `a`
/// and `b`, then horizontally sum each consecutive 8 differences to produce
/// two unsigned 16-bit integers, and pack these unsigned 16-bit integers in
/// the low 16 bits of 64-bit elements returned.
#[inline(always)]
pub unsafe fn _mm_sad_epu8(a: u8x16, b: u8x16) -> u64x2 {
    psadbw(a, b)
}

/// Subtract packed 8-bit integers in `b` from packed 8-bit integers in `a`,
/// and return the results.
#[inline(always)]
pub unsafe fn _mm_sub_epi8(a: i8x16, b: i8x16) -> i8x16 {
    simd_sub(a, b)
}

/// Subtract packed 16-bit integers in `b` from packed 16-bit integers in `a`,
/// and return the results.
#[inline(always)]
pub unsafe fn _mm_sub_epi16(a: i16x8, b: i16x8) -> i16x8 {
    simd_sub(a, b)
}

/// Subtract packed 32-bit integers in `b` from packed 32-bit integers in `a`,
/// and return the results.
#[inline(always)]
pub unsafe fn _mm_sub_epi32(a: i32x4, b: i32x4) -> i32x4 {
    simd_sub(a, b)
}

/// Subtract packed 64-bit integers in `b` from packed 64-bit integers in `a`,
/// and return the results.
#[inline(always)]
pub unsafe fn _mm_sub_epi64(a: i64x2, b: i64x2) -> i64x2 {
    simd_sub(a, b)
}

/// Subtract packed 8-bit integers in `b` from packed 8-bit integers in `a`
/// using saturation, and return the results.
#[inline(always)]
pub unsafe fn _mm_subs_epi8(a: i8x16, b: i8x16) -> i8x16 {
    psubsb(a, b)
}

/// Subtract packed 16-bit integers in `b` from packed 16-bit integers in `a`
/// using saturation, and return the results.
#[inline(always)]
pub unsafe fn _mm_subs_epi16(a: i16x8, b: i16x8) -> i16x8 {
    psubsw(a, b)
}

/// Subtract packed unsigned 8-bit integers in `b` from packed unsigned 8-bit
/// integers in `a` using saturation, and return the results.
#[inline(always)]
pub unsafe fn _mm_subs_epu8(a: u8x16, b: u8x16) -> u8x16 {
    psubusb(a, b)
}

/// Subtract packed unsigned 16-bit integers in `b` from packed unsigned 16-bit
/// integers in `a` using saturation, and return the results.
#[inline(always)]
pub unsafe fn _mm_subs_epu16(a: u16x8, b: u16x8) -> u16x8 {
    psubusw(a, b)
}

/// Shift `a` left by `imm8` bytes while shifting in zeros, and return the
/// results.
#[inline(always)]
pub unsafe fn _mm_slli_si128(a: __m128i, imm8: i32) -> __m128i {
    let (zero, imm8) = (__m128i::splat(0), imm8 as u32);
    const fn sub(a: u32, b: u32) -> u32 { a - b }
    macro_rules! shuffle {
        ($shift:expr) => {
            simd_shuffle16::<__m128i, __m128i>(zero, a, [
                sub(16, $shift), sub(17, $shift),
                sub(18, $shift), sub(19, $shift),
                sub(20, $shift), sub(21, $shift),
                sub(22, $shift), sub(23, $shift),
                sub(24, $shift), sub(25, $shift),
                sub(26, $shift), sub(27, $shift),
                sub(28, $shift), sub(29, $shift),
                sub(30, $shift), sub(31, $shift),
            ])
        }
    }
    match imm8 {
        0 => shuffle!(0), 1 => shuffle!(1),
        2 => shuffle!(2), 3 => shuffle!(3),
        4 => shuffle!(4), 5 => shuffle!(5),
        6 => shuffle!(6), 7 => shuffle!(7),
        8 => shuffle!(8), 9 => shuffle!(9),
        10 => shuffle!(10), 11 => shuffle!(11),
        12 => shuffle!(12), 13 => shuffle!(13),
        14 => shuffle!(14), 15 => shuffle!(15),
        _ => shuffle!(16),
    }
}

/// Shift `a` left by `imm8` bytes while shifting in zeros, and return the
/// results.
#[inline(always)]
pub unsafe fn _mm_bslli_si128(a: __m128i, imm8: i32) -> __m128i {
    _mm_slli_si128(a, imm8)
}

/// Shift `a` right by `imm8` bytes while shifting in zeros, and return the
/// results.
#[inline(always)]
pub unsafe fn _mm_bsrli_si128(a: __m128i, imm8: i32) -> __m128i {
    _mm_srli_si128(a, imm8)
}

/// Shift packed 16-bit integers in `a` left by `imm8` while shifting in zeros,
/// and return the results.
#[inline(always)]
pub unsafe fn _mm_slli_epi16(a: i16x8, imm8: i32) -> i16x8  {
    pslliw(a, imm8)
}


/// Shift `a` right by `imm8` bytes while shifting in zeros, and return the
/// results.
#[inline(always)]
pub unsafe fn _mm_srli_si128(a: __m128i, imm8: i32) -> __m128i {
    let (zero, imm8) = (__m128i::splat(0), imm8 as u32);
    const fn add(a: u32, b: u32) -> u32 { a + b }
    macro_rules! shuffle {
        ($shift:expr) => {
            simd_shuffle16::<__m128i, __m128i>(a, zero, [
                add(0, $shift), add(1, $shift),
                add(2, $shift), add(3, $shift),
                add(4, $shift), add(5, $shift),
                add(6, $shift), add(7, $shift),
                add(8, $shift), add(9, $shift),
                add(10, $shift), add(11, $shift),
                add(12, $shift), add(13, $shift),
                add(14, $shift), add(15, $shift),
            ])
        }
    }
    match imm8 {
        0 => shuffle!(0), 1 => shuffle!(1),
        2 => shuffle!(2), 3 => shuffle!(3),
        4 => shuffle!(4), 5 => shuffle!(5),
        6 => shuffle!(6), 7 => shuffle!(7),
        8 => shuffle!(8), 9 => shuffle!(9),
        10 => shuffle!(10), 11 => shuffle!(11),
        12 => shuffle!(12), 13 => shuffle!(13),
        14 => shuffle!(14), 15 => shuffle!(15),
        _ => shuffle!(16),
    }
}

/// Convert the lower two packed 32-bit integers in `a` to packed
/// double-precision (64-bit) floating-point elements, and return the results.
#[inline(always)]
pub unsafe fn _mm_cvtepi32_pd(a: i32x4) -> f64x2  {
    simd_cast::<i32x2, f64x2>(simd_shuffle2(a, a, [0, 1]))
}

/// Set packed 64-bit integers with the supplied values.
#[inline(always)]
pub unsafe fn _mm_set_epi64x(e1: i64, e0: i64) -> i64x2 {
    i64x2::new(e0, e1)
}

/// Set packed 32-bit integers with the supplied values.
#[inline(always)]
pub unsafe fn _mm_set_epi32(e3: i32, e2: i32, e1: i32, e0: i32) -> i32x4 {
    i32x4::new(e0, e1, e2, e3)
}

/// Set packed 16-bit integers with the supplied values.
#[inline(always)]
pub unsafe fn _mm_set_epi16(
    e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16,
) -> i16x8 {
    i16x8::new(e0, e1, e2, e3, e4, e5, e6, e7)
}

/// Set packed 8-bit integers with the supplied values.
#[inline(always)]
pub unsafe fn _mm_set_epi8(
    e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8,
    e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8,
) -> i8x16 {
    i8x16::new(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
    )
}

/// Broadcast 64-bit integer `a` to all elements.
#[inline(always)]
pub unsafe fn _mm_set1_epi64x(a: i64) -> i64x2 {
    i64x2::splat(a)
}

/// Broadcast 32-bit integer `a` to all elements.
#[inline(always)]
pub unsafe fn _mm_set1_epi32(a: i32) -> i32x4 {
    i32x4::splat(a)
}

/// Broadcast 16-bit integer `a` to all elements.
#[inline(always)]
pub unsafe fn _mm_set1_epi16(a: i16) -> i16x8 {
    i16x8::splat(a)
}

/// Broadcast 8-bit integer `a` to all elements.
#[inline(always)]
pub unsafe fn _mm_set1_epi8(a: i8) -> i8x16 {
    i8x16::splat(a)
}

/// Set packed 32-bit integers with the supplied values in reverse order.
#[inline(always)]
pub unsafe fn _mm_setr_epi32(e3: i32, e2: i32, e1: i32, e0: i32) -> i32x4 {
    i32x4::new(e3, e2, e1, e0)
}

/// Set packed 16-bit integers with the supplied values in reverse order.
#[inline(always)]
pub unsafe fn _mm_setr_epi16(
    e7: i16, e6: i16, e5: i16, e4: i16, e3: i16, e2: i16, e1: i16, e0: i16,
) -> i16x8 {
    i16x8::new(e7, e6, e5, e4, e3, e2, e1, e0)
}

/// Set packed 8-bit integers with the supplied values in reverse order.
#[inline(always)]
pub unsafe fn _mm_setr_epi8(
    e15: i8, e14: i8, e13: i8, e12: i8, e11: i8, e10: i8, e9: i8, e8: i8,
    e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8,
) -> i8x16 {
    i8x16::new(
        e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0,
    )
}

/// Returns a vector with all elements set to zero.
#[inline(always)]
pub unsafe fn _mm_setzero_si128() -> __m128i {
    __m128i::splat(0)
}

/// Load 64-bit integer from memory into first element of returned vector.
#[inline(always)]
pub unsafe fn _mm_loadl_epi64(mem_addr: *const i64x2) -> i64x2 {
    _mm_set_epi64x(0, (*mem_addr).extract(0))
}

/// Load 128-bits of integer data from memory into a new vector.
///
/// `mem_addr` must be aligned on a 16-byte boundary.
#[inline(always)]
pub unsafe fn _mm_load_si128(mem_addr: *const __m128i) -> __m128i {
    *mem_addr
}

/// Load 128-bits of integer data from memory into a new vector.
///
/// `mem_addr` does not need to be aligned on any particular boundary.
#[inline(always)]
pub unsafe fn _mm_loadu_si128(mem_addr: *const __m128i) -> __m128i {
    let mut dst = mem::uninitialized();
    ptr::copy_nonoverlapping(
        mem_addr as *const u8,
        &mut dst as *mut __m128i as *mut u8,
        mem::size_of::<__m128i>());
    dst
}









#[inline(always)]
pub unsafe fn _mm_add_sd(a: f64x2, b: f64x2) -> f64x2 {
    a.insert(0, a.extract(0) + b.extract(0))
}

#[inline(always)]
pub unsafe fn _mm_add_pd(a: f64x2, b: f64x2) -> f64x2 {
    simd_add(a, b)
}

#[inline(always)]
pub unsafe fn _mm_load_pd(mem_addr: *const f64) -> f64x2 {
    *(mem_addr as *const f64x2)
}

#[inline(always)]
pub unsafe fn _mm_store_pd(mem_addr: *mut f64, a: f64x2) {
    *(mem_addr as *mut f64x2) = a;
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
    #[link_name = "llvm.x86.sse2.psubs.b"]
    pub fn psubsb(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.x86.sse2.psubs.w"]
    pub fn psubsw(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.x86.sse2.psubus.b"]
    pub fn psubusb(a: u8x16, b: u8x16) -> u8x16;
    #[link_name = "llvm.x86.sse2.psubus.w"]
    pub fn psubusw(a: u16x8, b: u16x8) -> u16x8;
    #[link_name = "llvm.x86.sse2.pslli.w"]
    pub fn pslliw(a: i16x8, imm8: i32) -> i16x8;
}

#[cfg(test)]
mod tests {
    use std::os::raw::c_void;

    use v128::*;
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
        let a = i8x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = i8x16::new(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = unsafe { sse2::_mm_add_epi8(a, b) };
        let e = i8x16::new(
            16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46);
        assert_eq!(r, e);
    }

    #[test]
    fn _mm_add_epi8_overflow() {
        let a = i8x16::splat(0x7F);
        let b = i8x16::splat(1);
        let r = unsafe { sse2::_mm_add_epi8(a, b) };
        assert_eq!(r, i8x16::splat(-128));
    }

    #[test]
    fn _mm_add_epi16() {
        let a = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = i16x8::new(8, 9, 10, 11, 12, 13, 14, 15);
        let r = unsafe { sse2::_mm_add_epi16(a, b) };
        let e = i16x8::new(8, 10, 12, 14, 16, 18, 20, 22);
        assert_eq!(r, e);
    }

    #[test]
    fn _mm_add_epi32() {
        let a = i32x4::new(0, 1, 2, 3);
        let b = i32x4::new(4, 5, 6, 7);
        let r = unsafe { sse2::_mm_add_epi32(a, b) };
        let e = i32x4::new(4, 6, 8, 10);
        assert_eq!(r, e);
    }

    #[test]
    fn _mm_add_epi64() {
        let a = i64x2::new(0, 1);
        let b = i64x2::new(2, 3);
        let r = unsafe { sse2::_mm_add_epi64(a, b) };
        let e = i64x2::new(2, 4);
        assert_eq!(r, e);
    }

    #[test]
    fn _mm_adds_epi8() {
        let a = i8x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = i8x16::new(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = unsafe { sse2::_mm_adds_epi8(a, b) };
        let e = i8x16::new(
            16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46);
        assert_eq!(r, e);
    }

    #[test]
    fn _mm_adds_epi8_saturate_positive() {
        let a = i8x16::splat(0x7F);
        let b = i8x16::splat(1);
        let r = unsafe { sse2::_mm_adds_epi8(a, b) };
        assert_eq!(r, a);
    }

    #[test]
    fn _mm_adds_epi8_saturate_negative() {
        let a = i8x16::splat(-0x80);
        let b = i8x16::splat(-1);
        let r = unsafe { sse2::_mm_adds_epi8(a, b) };
        assert_eq!(r, a);
    }

    #[test]
    fn _mm_adds_epi16() {
        let a = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = i16x8::new(8, 9, 10, 11, 12, 13, 14, 15);
        let r = unsafe { sse2::_mm_adds_epi16(a, b) };
        let e = i16x8::new(8, 10, 12, 14, 16, 18, 20, 22);
        assert_eq!(r, e);
    }

    #[test]
    fn _mm_adds_epi16_saturate_positive() {
        let a = i16x8::splat(0x7FFF);
        let b = i16x8::splat(1);
        let r = unsafe { sse2::_mm_adds_epi16(a, b) };
        assert_eq!(r, a);
    }

    #[test]
    fn _mm_adds_epi16_saturate_negative() {
        let a = i16x8::splat(-0x8000);
        let b = i16x8::splat(-1);
        let r = unsafe { sse2::_mm_adds_epi16(a, b) };
        assert_eq!(r, a);
    }

    #[test]
    fn _mm_adds_epu8() {
        let a = u8x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = u8x16::new(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = unsafe { sse2::_mm_adds_epu8(a, b) };
        let e = u8x16::new(
            16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46);
        assert_eq!(r, e);
    }

    #[test]
    fn _mm_adds_epu8_saturate() {
        let a = u8x16::splat(0xFF);
        let b = u8x16::splat(1);
        let r = unsafe { sse2::_mm_adds_epu8(a, b) };
        assert_eq!(r, a);
    }

    #[test]
    fn _mm_adds_epu16() {
        let a = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b = u16x8::new(8, 9, 10, 11, 12, 13, 14, 15);
        let r = unsafe { sse2::_mm_adds_epu16(a, b) };
        let e = u16x8::new(8, 10, 12, 14, 16, 18, 20, 22);
        assert_eq!(r, e);
    }

    #[test]
    fn _mm_adds_epu16_saturate() {
        let a = u16x8::splat(0xFFFF);
        let b = u16x8::splat(1);
        let r = unsafe { sse2::_mm_adds_epu16(a, b) };
        assert_eq!(r, a);
    }

    #[test]
    fn _mm_avg_epu8() {
        let (a, b) = (u8x16::splat(3), u8x16::splat(9));
        let r = unsafe { sse2::_mm_avg_epu8(a, b) };
        assert_eq!(r, u8x16::splat(6));
    }

    #[test]
    fn _mm_avg_epu16() {
        let (a, b) = (u16x8::splat(3), u16x8::splat(9));
        let r = unsafe { sse2::_mm_avg_epu16(a, b) };
        assert_eq!(r, u16x8::splat(6));
    }

    #[test]
    fn _mm_madd_epi16() {
        let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b = i16x8::new(9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { sse2::_mm_madd_epi16(a, b) };
        let e = i32x4::new(29, 81, 149, 233);
        assert_eq!(r, e);
    }

    #[test]
    fn _mm_max_epi16() {
        let a = i16x8::splat(1);
        let b = i16x8::splat(-1);
        let r = unsafe { sse2::_mm_max_epi16(a, b) };
        assert_eq!(r, a);
    }

    #[test]
    fn _mm_max_epu8() {
        let a = u8x16::splat(1);
        let b = u8x16::splat(255);
        let r = unsafe { sse2::_mm_max_epu8(a, b) };
        assert_eq!(r, b);
    }

    #[test]
    fn _mm_min_epi16() {
        let a = i16x8::splat(1);
        let b = i16x8::splat(-1);
        let r = unsafe { sse2::_mm_min_epi16(a, b) };
        assert_eq!(r, b);
    }

    #[test]
    fn _mm_min_epu8() {
        let a = u8x16::splat(1);
        let b = u8x16::splat(255);
        let r = unsafe { sse2::_mm_min_epu8(a, b) };
        assert_eq!(r, a);
    }

    #[test]
    fn _mm_mulhi_epi16() {
        let (a, b) = (i16x8::splat(1000), i16x8::splat(-1001));
        let r = unsafe { sse2::_mm_mulhi_epi16(a, b) };
        assert_eq!(r, i16x8::splat(-16));
    }

    #[test]
    fn _mm_mulhi_epu16() {
        let (a, b) = (u16x8::splat(1000), u16x8::splat(1001));
        let r = unsafe { sse2::_mm_mulhi_epu16(a, b) };
        assert_eq!(r, u16x8::splat(15));
    }

    #[test]
    fn _mm_mullo_epi16() {
        let (a, b) = (i16x8::splat(1000), i16x8::splat(-1001));
        let r = unsafe { sse2::_mm_mullo_epi16(a, b) };
        assert_eq!(r, i16x8::splat(-17960));
    }

    #[test]
    fn _mm_mul_epu32() {
        let a = u32x4::from(u64x2::new(1_000_000_000, 1 << 34));
        let b = u32x4::from(u64x2::new(1_000_000_000, 1 << 35));
        let r = unsafe { sse2::_mm_mul_epu32(a, b) };
        let e = u64x2::new(1_000_000_000 * 1_000_000_000, 0);
        assert_eq!(r, e);
    }

    #[test]
    fn _mm_sad_epu8() {
        let a = u8x16::new(
            255, 254, 253, 252, 1, 2, 3, 4,
            155, 154, 153, 152, 1, 2, 3, 4);
        let b = u8x16::new(
            0, 0, 0, 0, 2, 1, 2, 1,
            1, 1, 1, 1, 1, 2, 1, 2);
        let r = unsafe { sse2::_mm_sad_epu8(a, b) };
        let e = u64x2::new(1020, 614);
        assert_eq!(r, e);
    }

    #[test]
    fn _mm_sub_epi8() {
        let (a, b) = (i8x16::splat(5), i8x16::splat(6));
        let r = unsafe { sse2::_mm_sub_epi8(a, b) };
        assert_eq!(r, i8x16::splat(-1));
    }

    #[test]
    fn _mm_sub_epi16() {
        let (a, b) = (i16x8::splat(5), i16x8::splat(6));
        let r = unsafe { sse2::_mm_sub_epi16(a, b) };
        assert_eq!(r, i16x8::splat(-1));
    }

    #[test]
    fn _mm_sub_epi32() {
        let (a, b) = (i32x4::splat(5), i32x4::splat(6));
        let r = unsafe { sse2::_mm_sub_epi32(a, b) };
        assert_eq!(r, i32x4::splat(-1));
    }

    #[test]
    fn _mm_sub_epi64() {
        let (a, b) = (i64x2::splat(5), i64x2::splat(6));
        let r = unsafe { sse2::_mm_sub_epi64(a, b) };
        assert_eq!(r, i64x2::splat(-1));
    }

    #[test]
    fn _mm_subs_epi8() {
        let (a, b) = (i8x16::splat(5), i8x16::splat(2));
        let r = unsafe { sse2::_mm_subs_epi8(a, b) };
        assert_eq!(r, i8x16::splat(3));
    }

    #[test]
    fn _mm_subs_epi8_saturate_positive() {
        let a = i8x16::splat(0x7F);
        let b = i8x16::splat(-1);
        let r = unsafe { sse2::_mm_subs_epi8(a, b) };
        assert_eq!(r, a);
    }

    #[test]
    fn _mm_subs_epi8_saturate_negative() {
        let a = i8x16::splat(-0x80);
        let b = i8x16::splat(1);
        let r = unsafe { sse2::_mm_subs_epi8(a, b) };
        assert_eq!(r, a);
    }

    #[test]
    fn _mm_subs_epi16() {
        let (a, b) = (i16x8::splat(5), i16x8::splat(2));
        let r = unsafe { sse2::_mm_subs_epi16(a, b) };
        assert_eq!(r, i16x8::splat(3));
    }

    #[test]
    fn _mm_subs_epi16_saturate_positive() {
        let a = i16x8::splat(0x7FFF);
        let b = i16x8::splat(-1);
        let r = unsafe { sse2::_mm_subs_epi16(a, b) };
        assert_eq!(r, a);
    }

    #[test]
    fn _mm_subs_epi16_saturate_negative() {
        let a = i16x8::splat(-0x8000);
        let b = i16x8::splat(1);
        let r = unsafe { sse2::_mm_subs_epi16(a, b) };
        assert_eq!(r, a);
    }

    #[test]
    fn _mm_subs_epu8() {
        let (a, b) = (u8x16::splat(5), u8x16::splat(2));
        let r = unsafe { sse2::_mm_subs_epu8(a, b) };
        assert_eq!(r, u8x16::splat(3));
    }

    #[test]
    fn _mm_subs_epu8_saturate() {
        let a = u8x16::splat(0);
        let b = u8x16::splat(1);
        let r = unsafe { sse2::_mm_subs_epu8(a, b) };
        assert_eq!(r, a);
    }

    #[test]
    fn _mm_subs_epu16() {
        let (a, b) = (u16x8::splat(5), u16x8::splat(2));
        let r = unsafe { sse2::_mm_subs_epu16(a, b) };
        assert_eq!(r, u16x8::splat(3));
    }

    #[test]
    fn _mm_subs_epu16_saturate() {
        let a = u16x8::splat(0);
        let b = u16x8::splat(1);
        let r = unsafe { sse2::_mm_subs_epu16(a, b) };
        assert_eq!(r, a);
    }

    #[test]
    fn _mm_slli_si128() {
        let a = __m128i::new(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { sse2::_mm_slli_si128(a, 1) };
        let e = __m128i::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq!(r, e);

        let a = __m128i::new(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { sse2::_mm_slli_si128(a, 15) };
        let e = __m128i::new(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
        assert_eq!(r, e);

        let a = __m128i::new(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { sse2::_mm_slli_si128(a, 16) };
        assert_eq!(r, __m128i::splat(0));

        let a = __m128i::new(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { sse2::_mm_slli_si128(a, -1) };
        assert_eq!(r, __m128i::splat(0));

        let a = __m128i::new(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { sse2::_mm_slli_si128(a, -0x80000000) };
        assert_eq!(r, __m128i::splat(0));
    }

    #[test]
    fn _mm_slli_epi16() {
        let a = i16x8::new(
            0xFFFF as u16 as i16, 0x0FFF, 0x00FF, 0x000F, 0, 0, 0, 0);
        let r = unsafe { sse2::_mm_slli_epi16(a, 4) };
        let e = i16x8::new(
            0xFFF0 as u16 as i16,
            0xFFF0 as u16 as i16, 0x0FF0, 0x00F0, 0, 0, 0, 0);
        assert_eq!(r, e);
    }

    #[test]
    fn _mm_srli_si128() {
        let a = __m128i::new(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { sse2::_mm_srli_si128(a, 1) };
        let e = __m128i::new(
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0);
        assert_eq!(r, e);

        let a = __m128i::new(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { sse2::_mm_srli_si128(a, 15) };
        let e = __m128i::new(
            16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq!(r, e);

        let a = __m128i::new(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { sse2::_mm_srli_si128(a, 16) };
        assert_eq!(r, __m128i::splat(0));

        let a = __m128i::new(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { sse2::_mm_srli_si128(a, -1) };
        assert_eq!(r, __m128i::splat(0));

        let a = __m128i::new(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { sse2::_mm_srli_si128(a, -0x80000000) };
        assert_eq!(r, __m128i::splat(0));
    }

    #[test]
    fn _mm_cvtepi32_pd() {
        unsafe {
            let a = sse2::_mm_set_epi32(35, 25, 15, 5);
            let r = sse2::_mm_cvtepi32_pd(a);
            assert_eq!(r, f64x2::new(5.0, 15.0));
        }
    }

    #[test]
    fn _mm_loadl_epi64() {
        unsafe {
            let a = sse2::_mm_set_epi64x(5, 6);
            let r = sse2::_mm_loadl_epi64(&a as *const _);
            assert_eq!(r, i64x2::new(6, 0));
        }
    }

    #[test]
    fn _mm_load_si128() {
        unsafe {
            let a = sse2::_mm_set_epi64x(5, 6);
            let r = sse2::_mm_load_si128(&a as *const _ as *const _);
            assert_eq!(a, i64x2::from(r));
        }
    }

    #[test]
    fn _mm_loadu_si128() {
        unsafe {
            let a = sse2::_mm_set_epi64x(5, 6);
            let r = sse2::_mm_loadu_si128(&a as *const _ as *const _);
            assert_eq!(a, i64x2::from(r));
        }
    }
}
