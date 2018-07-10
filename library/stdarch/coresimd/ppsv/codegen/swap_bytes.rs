//! Horizontal mask reductions.

#![allow(unused)]

use coresimd::simd::*;

pub trait SwapBytes {
    unsafe fn swap_bytes(self) -> Self;
}

// TODO: switch to shuffle API once it lands
// TODO: investigate `llvm.bswap`
macro_rules! impl_swap_bytes {
    (v16, $($id:ident,)+) => {$(
        impl SwapBytes for $id {
            #[inline]
            unsafe fn swap_bytes(self) -> Self {
                use coresimd::simd_llvm::simd_shuffle2;

                const INDICES: [u32; 2] = [1, 0];
                simd_shuffle2(self, self, INDICES)
            }
        }
    )+};
    (v32, $($id:ident,)+) => {$(
        impl SwapBytes for $id {
            #[inline]
            unsafe fn swap_bytes(self) -> Self {
                use coresimd::simd_llvm::simd_shuffle4;

                const INDICES: [u32; 4] = [3, 2, 1, 0];
                let vec8 = u8x4::from_bits(self);
                let shuffled: u8x4 = simd_shuffle4(vec8, vec8, INDICES);
                $id::from_bits(shuffled)
            }
        }
    )+};
    (v64, $($id:ident,)+) => {$(
        impl SwapBytes for $id {
            #[inline]
            unsafe fn swap_bytes(self) -> Self {
                use coresimd::simd_llvm::simd_shuffle8;

                const INDICES: [u32; 8] = [7, 6, 5, 4, 3, 2, 1, 0];
                let vec8 = u8x8::from_bits(self);
                let shuffled: u8x8 = simd_shuffle8(vec8, vec8, INDICES);
                $id::from_bits(shuffled)
            }
        }
    )+};
    (v128, $($id:ident,)+) => {$(
        impl SwapBytes for $id {
            #[inline]
            unsafe fn swap_bytes(self) -> Self {
                use coresimd::simd_llvm::simd_shuffle16;

                const INDICES: [u32; 16] = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
                let vec8 = u8x16::from_bits(self);
                let shuffled: u8x16 = simd_shuffle16(vec8, vec8, INDICES);
                $id::from_bits(shuffled)
            }
        }
    )+};
    (v256, $($id:ident,)+) => {$(
        impl SwapBytes for $id {
            #[inline]
            unsafe fn swap_bytes(self) -> Self {
                use coresimd::simd_llvm::simd_shuffle32;

                const INDICES: [u32; 32] = [
                    31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
                    15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
                ];
                let vec8 = u8x32::from_bits(self);
                let shuffled: u8x32 = simd_shuffle32(vec8, vec8, INDICES);
                $id::from_bits(shuffled)
            }
        }
    )+};
    (v512, $($id:ident,)+) => {$(
        impl SwapBytes for $id {
            #[inline]
            unsafe fn swap_bytes(self) -> Self {
                use coresimd::simd_llvm::simd_shuffle64;

                const INDICES: [u32; 64] = [
                    63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
                    47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
                    31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
                    15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0,
                ];
                let vec8 = u8x64::from_bits(self);
                let shuffled: u8x64 = simd_shuffle64(vec8, vec8, INDICES);
                $id::from_bits(shuffled)
            }
        }
    )+};
}

vector_impl!(
    [impl_swap_bytes, v16, u8x2, i8x2,],
    [impl_swap_bytes, v32, u8x4, i8x4, u16x2, i16x2,],
    [impl_swap_bytes, v64, u8x8, i8x8, u16x4, i16x4, u32x2, i32x2,],
    [
        impl_swap_bytes,
        v128,
        u8x16,
        i8x16,
        u16x8,
        i16x8,
        u32x4,
        i32x4,
        u64x2,
        i64x2,
    ],
    [
        impl_swap_bytes,
        v256,
        u8x32,
        i8x32,
        u16x16,
        i16x16,
        u32x8,
        i32x8,
        u64x4,
        i64x4,
    ],
    [
        impl_swap_bytes,
        v512,
        u8x64,
        i8x64,
        u16x32,
        i16x32,
        u32x16,
        i32x16,
        u64x8,
        i64x8,
    ]
);
