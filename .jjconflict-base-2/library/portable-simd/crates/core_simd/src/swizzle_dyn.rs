// Allow these FCW: anyone soundly using the intrinsics has to enable
// the target feature, and that will generate a warning for them.
#![allow(aarch64_softfloat_neon, x86_softfloat_sse)]

use crate::simd::Simd;
use core::mem;

impl<const N: usize> Simd<u8, N> {
    /// Swizzle a vector of bytes according to the index vector.
    /// Indices within range select the appropriate byte.
    /// Indices "out of bounds" instead select 0.
    ///
    /// Note that the current implementation is selected during build-time
    /// of the standard library, so `cargo build -Zbuild-std` may be necessary
    /// to unlock better performance, especially for larger vectors.
    /// A planned compiler improvement will enable using `#[target_feature]` instead.
    #[inline]
    pub fn swizzle_dyn(self, idxs: Simd<u8, N>) -> Self {
        #![allow(unused_imports, unused_unsafe)]
        #[cfg(all(
            target_arch = "arm",
            target_feature = "v7",
            target_feature = "neon",
            target_endian = "little"
        ))]
        use core::arch::arm::{uint8x8_t, vtbl1_u8};
        #[cfg(target_arch = "wasm32")]
        use core::arch::wasm32 as wasm;
        #[cfg(target_arch = "wasm64")]
        use core::arch::wasm64 as wasm;
        #[cfg(target_arch = "x86")]
        use core::arch::x86;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64 as x86;
        // SAFETY: Intrinsics covered by cfg
        unsafe {
            #[allow(
                unreachable_patterns,
                reason = "avoids writing verbose cfg(not), earlier branches take priority"
            )]
            match N {
                // Aarch64
                #[cfg(all(
                    any(target_arch = "aarch64", target_arch = "arm64ec"),
                    target_feature = "neon",
                    target_endian = "little"
                ))]
                8 | 16 | 24 | 32 | 48 | 64 => aarch64_swizzle(self, idxs),

                // 32-bit ARMv7
                #[cfg(all(
                    target_arch = "arm",
                    target_feature = "v7",
                    target_feature = "neon",
                    target_endian = "little"
                ))]
                16 => transize(armv7_neon_swizzle_u8x16, self, idxs),

                // WASM SIMD128
                #[cfg(target_feature = "simd128")]
                16 => transize(wasm::i8x16_swizzle, self, idxs),
                #[cfg(target_feature = "simd128")]
                32 => transize(swizzle_dyn_split::<32, 16>, self, idxs),

                // LoongArch64
                #[cfg(all(target_arch = "loongarch64", target_feature = "lsx"))]
                16 => transize(loong64_lsx_swizzle, self, idxs),
                #[cfg(all(target_arch = "loongarch64", target_feature = "lasx"))]
                32 => transize(loong64_lasx_swizzle, self, idxs),
                #[cfg(all(target_arch = "loongarch64", target_feature = "lsx"))]
                32 => transize(swizzle_dyn_split::<32, 16>, self, idxs),
                #[cfg(all(target_arch = "loongarch64", target_feature = "lasx"))]
                64 => transize(swizzle_dyn_split::<64, 32>, self, idxs),

                // x86, x86-64
                #[cfg(target_feature = "ssse3")]
                16 => transize(x86::_mm_shuffle_epi8, self, zeroing_idxs(idxs)),
                #[cfg(all(target_feature = "avx512vl", target_feature = "avx512vbmi"))]
                32 => {
                    // Unlike vpshufb, vpermb doesn't zero out values in the result based on the index high bit
                    let swizzler = |bytes, idxs| {
                        let mask = x86::_mm256_cmp_epu8_mask::<{ x86::_MM_CMPINT_LT }>(
                            idxs,
                            Simd::<u8, 32>::splat(N as u8).into(),
                        );
                        x86::_mm256_maskz_permutexvar_epi8(mask, idxs, bytes)
                    };
                    transize(swizzler, self, idxs)
                }
                #[cfg(target_feature = "avx2")]
                32 => transize(avx2_pshufb, self, idxs),
                #[cfg(target_feature = "ssse3")]
                32 => transize(swizzle_dyn_split::<32, 16>, self, idxs),
                // Notable absence: avx512bw pshufb shuffle
                #[cfg(all(target_feature = "avx512vl", target_feature = "avx512vbmi"))]
                64 => {
                    // Unlike vpshufb, vpermb doesn't zero out values in the result based on the index high bit
                    let swizzler = |bytes, idxs| {
                        let mask = x86::_mm512_cmp_epu8_mask::<{ x86::_MM_CMPINT_LT }>(
                            idxs,
                            Simd::<u8, 64>::splat(N as u8).into(),
                        );
                        x86::_mm512_maskz_permutexvar_epi8(mask, idxs, bytes)
                    };
                    transize(swizzler, self, idxs)
                }
                #[cfg(target_feature = "avx2")]
                64 => transize(swizzle_dyn_split::<64, 32>, self, idxs),

                // scalar fallback
                _ => {
                    let mut array = [0; N];
                    for (i, k) in idxs.to_array().into_iter().enumerate() {
                        if (k as usize) < N {
                            array[i] = self[k as usize];
                        };
                    }
                    array.into()
                }
            }
        }
    }
}

#[allow(dead_code, reason = "only used on some targets/features")]
/// Implements an arbitrary shuffle over double the native vector width
/// using 4 native-width shuffles
fn swizzle_dyn_split<const N: usize, const HALF: usize>(
    bytes: Simd<u8, N>,
    idxs: Simd<u8, N>,
) -> Simd<u8, N> {
    let table_low = bytes.extract::<0, HALF>();
    let table_high = bytes.extract::<HALF, HALF>();
    let idxs_low = idxs.extract::<0, HALF>();
    let idxs_high = idxs.extract::<HALF, HALF>();
    let table_high_offset = Simd::<u8, HALF>::splat(HALF as u8);

    let output_low_from_low = table_low.swizzle_dyn(idxs_low);
    let output_low_from_high = table_high.swizzle_dyn(idxs_low - table_high_offset);
    let output_low = output_low_from_low | output_low_from_high;

    let output_high_from_low = table_low.swizzle_dyn(idxs_high);
    let output_high_from_high = table_high.swizzle_dyn(idxs_high - table_high_offset);
    let output_high = output_high_from_low | output_high_from_high;

    // This is simply a concatenation of two native-sized vectors.
    // The swizzle does nothing - it maps the elements right back where they already are.
    // There doesn't seem to be a more direct way to do this as of this writing.
    // TODO: simplify once a plain `concat` is available.
    use crate::simd::Swizzle;
    struct CombineHalves;
    impl<const N: usize> Swizzle<N> for CombineHalves {
        const INDEX: [usize; N] = const {
            let mut index = [0; N];
            let mut i = 0;
            while i < N {
                index[i] = i;
                i += 1;
            }
            index
        };
    }

    CombineHalves::concat_swizzle(output_low, output_high)
}

/// armv7 neon supports swizzling `u8x16` by swizzling two u8x8 blocks
/// with a u8x8x2 lookup table.
///
/// # Safety
/// This requires armv7 neon to work
#[cfg(all(
    target_arch = "arm",
    target_feature = "v7",
    target_feature = "neon",
    target_endian = "little"
))]
unsafe fn armv7_neon_swizzle_u8x16(bytes: Simd<u8, 16>, idxs: Simd<u8, 16>) -> Simd<u8, 16> {
    use core::arch::arm::{uint8x8x2_t, vcombine_u8, vget_high_u8, vget_low_u8, vtbl2_u8};
    // SAFETY: Caller promised arm neon support
    unsafe {
        let bytes = uint8x8x2_t(vget_low_u8(bytes.into()), vget_high_u8(bytes.into()));
        let lo = vtbl2_u8(bytes, vget_low_u8(idxs.into()));
        let hi = vtbl2_u8(bytes, vget_high_u8(idxs.into()));
        vcombine_u8(lo, hi).into()
    }
}

/// AArch64 NEON supports swizzling 8, 16, 24, 32, 48 or 64 by stacking multiple TBL instructions.
///
/// # Safety
/// This requires AArch64 NEON to work
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm64ec"),
    target_feature = "neon",
    target_endian = "little"
))]
unsafe fn aarch64_swizzle<const N: usize>(bytes: Simd<u8, N>, idxs: Simd<u8, N>) -> Simd<u8, N> {
    use core::arch::aarch64::*;
    use core::mem::transmute_copy;

    // SAFETY: Caller promised AArch64 NEON support
    unsafe {
        match N {
            8 => transmute_copy(&vtbl1_u8(transmute_copy(&bytes), transmute_copy(&idxs))),
            16 => transmute_copy(&vqtbl1q_u8(transmute_copy(&bytes), transmute_copy(&idxs))),
            24 => {
                let bytes: uint8x8x3_t = transmute_copy(&bytes);
                let idxs: uint8x8x3_t = transmute_copy(&idxs);

                let ret0 = vtbl3_u8(bytes, idxs.0);
                let ret1 = vtbl3_u8(bytes, idxs.1);
                let ret2 = vtbl3_u8(bytes, idxs.2);

                let ret = uint8x8x3_t(ret0, ret1, ret2);
                transmute_copy(&ret)
            }
            32 => {
                let bytes: uint8x16x2_t = transmute_copy(&bytes);
                let idxs: uint8x16x2_t = transmute_copy(&idxs);

                let ret0 = vqtbl2q_u8(bytes, idxs.0);
                let ret1 = vqtbl2q_u8(bytes, idxs.1);

                let ret = uint8x16x2_t(ret0, ret1);
                transmute_copy(&ret)
            }
            48 => {
                let bytes: uint8x16x3_t = transmute_copy(&bytes);
                let idxs: uint8x16x3_t = transmute_copy(&idxs);

                let ret0 = vqtbl3q_u8(bytes, idxs.0);
                let ret1 = vqtbl3q_u8(bytes, idxs.1);
                let ret2 = vqtbl3q_u8(bytes, idxs.2);

                let ret = uint8x16x3_t(ret0, ret1, ret2);
                transmute_copy(&ret)
            }
            64 => {
                let bytes: uint8x16x4_t = transmute_copy(&bytes);
                let idxs: uint8x16x4_t = transmute_copy(&idxs);

                let ret0 = vqtbl4q_u8(bytes, idxs.0);
                let ret1 = vqtbl4q_u8(bytes, idxs.1);
                let ret2 = vqtbl4q_u8(bytes, idxs.2);
                let ret3 = vqtbl4q_u8(bytes, idxs.3);

                let ret = uint8x16x4_t(ret0, ret1, ret2, ret3);
                transmute_copy(&ret)
            }
            _ => unreachable!(),
        }
    }
}

/// "vpshufb like it was meant to be" on AVX2
///
/// # Safety
/// This requires AVX2 to work
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unused)]
#[inline]
#[allow(clippy::let_and_return)]
unsafe fn avx2_pshufb(bytes: Simd<u8, 32>, idxs: Simd<u8, 32>) -> Simd<u8, 32> {
    #[cfg(target_arch = "x86")]
    use core::arch::x86;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64 as x86;
    use x86::_mm256_permute2x128_si256 as avx2_cross_shuffle;
    use x86::_mm256_shuffle_epi8 as avx2_half_pshufb;
    // SAFETY: Caller promised AVX2
    unsafe {
        let lolo = avx2_cross_shuffle::<0x00>(bytes.into(), bytes.into());
        let hihi = avx2_cross_shuffle::<0x11>(bytes.into(), bytes.into());

        // Adding 0x60 preserves the low nibble and bit 4 for valid
        // indices 0..=31. Larger indices get their high bit set, so
        // VPSHUFB supplies the required out-of-bounds zeroing.
        let control = x86::_mm256_adds_epu8(idxs.into(), x86::_mm256_set1_epi8(0x60));

        // Move index bit 4 into each byte's sign bit for VPBLENDVB.
        let select_high = x86::_mm256_slli_epi16::<3>(control);
        let from_low = avx2_half_pshufb(lolo, control);
        let from_high = avx2_half_pshufb(hihi, control);
        x86::_mm256_blendv_epi8(from_low, from_high, select_high).into()
    }
}

/// LoongArch64 LSX supports swizzling `u8x16`
///
/// # Safety
/// This requires LoongArch LSX to work
#[cfg(all(target_arch = "loongarch64", target_feature = "lsx"))]
unsafe fn loong64_lsx_swizzle(bytes: Simd<u8, 16>, idxs: Simd<u8, 16>) -> Simd<u8, 16> {
    use core::arch::loongarch64::{lsx_vand_v, lsx_vshuf_b, lsx_vslei_bu};
    // SAFETY: Caller promised loongarch lsx support
    unsafe {
        let bytes = lsx_vshuf_b(bytes.into(), bytes.into(), idxs.into());
        let mask = lsx_vslei_bu::<15>(idxs.into());
        lsx_vand_v(bytes, mask).into()
    }
}

/// LoongArch64 LASX supports swizzling `u8x32`
///
/// # Safety
/// This requires LoongArch LASX to work
#[cfg(all(target_arch = "loongarch64", target_feature = "lasx"))]
unsafe fn loong64_lasx_swizzle(bytes: Simd<u8, 32>, idxs: Simd<u8, 32>) -> Simd<u8, 32> {
    use core::arch::loongarch64::{lasx_xvand_v, lasx_xvpermi_q, lasx_xvshuf_b, lasx_xvslei_bu};
    // SAFETY: Caller promised loongarch lasx support
    unsafe {
        let lolo = lasx_xvpermi_q::<0x00>(bytes.into(), bytes.into());
        let hihi = lasx_xvpermi_q::<0x11>(bytes.into(), bytes.into());
        let bytes = lasx_xvshuf_b(hihi, lolo, idxs.into());
        let mask = lasx_xvslei_bu::<31>(idxs.into());
        lasx_xvand_v(bytes, mask).into()
    }
}

/// This sets up a call to an architecture-specific function, and in doing so
/// it persuades rustc that everything is the correct size. Which it is.
/// This would not be needed if one could convince Rust that, by matching on N,
/// N is that value, and thus it would be valid to substitute e.g. 16.
///
/// # Safety
/// The correctness of this function hinges on the sizes agreeing in actuality.
#[allow(dead_code)]
#[inline(always)]
unsafe fn transize<T, const N: usize>(
    f: unsafe fn(T, T) -> T,
    a: Simd<u8, N>,
    b: Simd<u8, N>,
) -> Simd<u8, N> {
    // SAFETY: Same obligation to use this function as to use mem::transmute_copy.
    unsafe { mem::transmute_copy(&f(mem::transmute_copy(&a), mem::transmute_copy(&b))) }
}

/// Make indices that yield 0 for x86
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unused)]
#[inline(always)]
fn zeroing_idxs<const N: usize>(idxs: Simd<u8, N>) -> Simd<u8, N> {
    // Adding this sets the high bit for indices N..=127, while PSHUFB ignores
    // the other changed bits. The OR preserves the high bit for indices 128..=255.
    let zeroing_bits = idxs + Simd::splat((127 - N + 1) as u8);
    idxs | zeroing_bits
}
