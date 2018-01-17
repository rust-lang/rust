//! `i686`'s Streaming SIMD Extensions 4a (`SSE4a`)

use core::mem;
use v128::*;

#[cfg(test)]
use stdsimd_test::assert_instr;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.sse4a.extrq"]
    fn extrq(x: i64x2, y: i8x16) -> i64x2;
    #[link_name = "llvm.x86.sse4a.insertq"]
    fn insertq(x: i64x2, y: i64x2) -> i64x2;
    #[link_name = "llvm.x86.sse4a.movnt.sd"]
    fn movntsd(x: *mut f64, y: f64x2);
    #[link_name = "llvm.x86.sse4a.movnt.ss"]
    fn movntss(x: *mut f32, y: f32x4);
}

// FIXME(blocked on #248): _mm_extracti_si64(x, len, idx) // EXTRQ
// FIXME(blocked on #248): _mm_inserti_si64(x, y, len, idx) // INSERTQ

/// Extracts the bit range specified by `y` from the lower 64 bits of `x`.
///
/// The [13:8] bits of `y` specify the index of the bit-range to extract. The
/// [5:0] bits of `y` specify the length of the bit-range to extract. All other
/// bits are ignored.
///
/// If the length is zero, it is interpreted as `64`. If the length and index
/// are zero, the lower 64 bits of `x` are extracted.
///
/// If `length == 0 && index > 0` or `lenght + index > 64` the result is
/// undefined.
#[inline(always)]
#[target_feature(enable = "sse4a")]
#[cfg_attr(test, assert_instr(extrq))]
pub unsafe fn _mm_extract_si64(x: i64x2, y: i64x2) -> i64x2 {
    extrq(x, mem::transmute(y))
}

/// Inserts the `[length:0]` bits of `y` into `x` at `index`.
///
/// The bits of `y`:
///
/// - `[69:64]` specify the `length`,
/// - `[77:72]` specify the index.
///
/// If the `length` is zero it is interpreted as `64`. If `index + length > 64`
/// or `index > 0 && length == 0` the result is undefined.
#[inline(always)]
#[target_feature(enable = "sse4a")]
#[cfg_attr(test, assert_instr(insertq))]
pub unsafe fn _mm_insert_si64(x: i64x2, y: i64x2) -> i64x2 {
    insertq(x, y)
}

/// Non-temporal store of `a.0` into `p`.
#[inline(always)]
#[target_feature(enable = "sse4a")]
#[cfg_attr(test, assert_instr(movntsd))]
pub unsafe fn _mm_stream_sd(p: *mut f64, a: f64x2) {
    movntsd(p, a);
}

/// Non-temporal store of `a.0` into `p`.
#[inline(always)]
#[target_feature(enable = "sse4a")]
#[cfg_attr(test, assert_instr(movntss))]
pub unsafe fn _mm_stream_ss(p: *mut f32, a: f32x4) {
    movntss(p, a);
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;
    use x86::i686::sse4a;
    use v128::*;

    #[simd_test = "sse4a"]
    unsafe fn _mm_extract_si64() {
        let b = 0b0110_0000_0000_i64;
        //        ^^^^ bit range extracted
        let x = i64x2::new(b, 0);
        let v = 0b001000___00___000100_i64;
        //        ^idx: 2^3 = 8 ^length = 2^2 = 4
        let y = i64x2::new(v, 0);
        let e = i64x2::new(0b0110_i64, 0);
        let r = sse4a::_mm_extract_si64(x, y);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4a"]
    unsafe fn _mm_insert_si64() {
        let i = 0b0110_i64;
        //        ^^^^ bit range inserted
        let z = 0b1010_1010_1010i64;
        //        ^^^^ bit range replaced
        let e = 0b0110_1010_1010i64;
        //        ^^^^ replaced 1010 with 0110
        let x = i64x2::new(z, 0);
        let expected = i64x2::new(e, 0);
        let v = 0b001000___00___000100_i64;
        //        ^idx: 2^3 = 8 ^length = 2^2 = 4
        let y = i64x2::new(i, v);
        let r = sse4a::_mm_insert_si64(x, y);
        assert_eq!(r, expected);
    }

    #[repr(align(16))]
    struct MemoryF64 {
        data: [f64; 2],
    }

    #[simd_test = "sse4a"]
    unsafe fn _mm_stream_sd() {
        let mut mem = MemoryF64 {
            data: [1.0_f64, 2.0],
        };
        {
            let vals = &mut mem.data;
            let d = vals.as_mut_ptr();

            let x = f64x2::new(3.0, 4.0);

            sse4a::_mm_stream_sd(d, x);
        }
        assert_eq!(mem.data[0], 3.0);
        assert_eq!(mem.data[1], 2.0);
    }

    #[repr(align(16))]
    struct MemoryF32 {
        data: [f32; 4],
    }

    #[simd_test = "sse4a"]
    unsafe fn _mm_stream_ss() {
        let mut mem = MemoryF32 {
            data: [1.0_f32, 2.0, 3.0, 4.0],
        };
        {
            let vals = &mut mem.data;
            let d = vals.as_mut_ptr();

            let x = f32x4::new(5.0, 6.0, 7.0, 8.0);

            sse4a::_mm_stream_ss(d, x);
        }
        assert_eq!(mem.data[0], 5.0);
        assert_eq!(mem.data[1], 2.0);
        assert_eq!(mem.data[2], 3.0);
        assert_eq!(mem.data[3], 4.0);
    }
}
