use crate::simd::{LaneCount, Simd, SupportedLaneCount};
use core::mem;

impl<const N: usize> Simd<u8, N>
where
    LaneCount<N>: SupportedLaneCount,
{
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
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        use core::arch::aarch64::{uint8x8_t, vqtbl1q_u8, vtbl1_u8};
        #[cfg(all(
            target_arch = "arm",
            target_feature = "v7",
            target_feature = "neon",
            target_endian = "little"
        ))]
        use core::arch::arm::{uint8x8_t, vtbl1_u8};
        #[cfg(target_arch = "wasm32")]
        use core::arch::wasm32 as wasm;
        #[cfg(target_arch = "x86")]
        use core::arch::x86;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64 as x86;
        // SAFETY: Intrinsics covered by cfg
        unsafe {
            match N {
                #[cfg(all(
                    any(
                        target_arch = "aarch64",
                        all(target_arch = "arm", target_feature = "v7")
                    ),
                    target_feature = "neon",
                    target_endian = "little"
                ))]
                8 => transize(vtbl1_u8, self, idxs),
                #[cfg(target_feature = "ssse3")]
                16 => transize(x86::_mm_shuffle_epi8, self, idxs),
                #[cfg(target_feature = "simd128")]
                16 => transize(wasm::i8x16_swizzle, self, idxs),
                #[cfg(all(
                    target_arch = "aarch64",
                    target_feature = "neon",
                    target_endian = "little"
                ))]
                16 => transize(vqtbl1q_u8, self, idxs),
                #[cfg(all(target_feature = "avx2", not(target_feature = "avx512vbmi")))]
                32 => transize_raw(avx2_pshufb, self, idxs),
                #[cfg(target_feature = "avx512vl,avx512vbmi")]
                32 => transize(x86::_mm256_permutexvar_epi8, self, idxs),
                // Notable absence: avx512bw shuffle
                // If avx512bw is available, odds of avx512vbmi are good
                // FIXME: initial AVX512VBMI variant didn't actually pass muster
                // #[cfg(target_feature = "avx512vbmi")]
                // 64 => transize(x86::_mm512_permutexvar_epi8, self, idxs),
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
    use crate::simd::SimdPartialOrd;
    #[cfg(target_arch = "x86")]
    use core::arch::x86;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64 as x86;
    use x86::_mm256_permute2x128_si256 as avx2_cross_shuffle;
    use x86::_mm256_shuffle_epi8 as avx2_half_pshufb;
    let mid = Simd::splat(16u8);
    let high = mid + mid;
    // SAFETY: Caller promised AVX2
    unsafe {
        // This is ordering sensitive, and LLVM will order these how you put them.
        // Most AVX2 impls use ~5 "ports", and only 1 or 2 are capable of permutes.
        // But the "compose" step will lower to ops that can also use at least 1 other port.
        // So this tries to break up permutes so composition flows through "open" ports.
        // Comparative benches should be done on multiple AVX2 CPUs before reordering this

        let hihi = avx2_cross_shuffle::<0x11>(bytes.into(), bytes.into());
        let hi_shuf = Simd::from(avx2_half_pshufb(
            hihi,        // duplicate the vector's top half
            idxs.into(), // so that using only 4 bits of an index still picks bytes 16-31
        ));
        // A zero-fill during the compose step gives the "all-Neon-like" OOB-is-0 semantics
        let compose = idxs.simd_lt(high).select(hi_shuf, Simd::splat(0));
        let lolo = avx2_cross_shuffle::<0x00>(bytes.into(), bytes.into());
        let lo_shuf = Simd::from(avx2_half_pshufb(lolo, idxs.into()));
        // Repeat, then pick indices < 16, overwriting indices 0-15 from previous compose step
        let compose = idxs.simd_lt(mid).select(lo_shuf, compose);
        compose
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
    bytes: Simd<u8, N>,
    idxs: Simd<u8, N>,
) -> Simd<u8, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let idxs = zeroing_idxs(idxs);
    // SAFETY: Same obligation to use this function as to use mem::transmute_copy.
    unsafe { mem::transmute_copy(&f(mem::transmute_copy(&bytes), mem::transmute_copy(&idxs))) }
}

/// Make indices that yield 0 for this architecture
#[inline(always)]
fn zeroing_idxs<const N: usize>(idxs: Simd<u8, N>) -> Simd<u8, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    // On x86, make sure the top bit is set.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let idxs = {
        use crate::simd::SimdPartialOrd;
        idxs.simd_lt(Simd::splat(N as u8))
            .select(idxs, Simd::splat(u8::MAX))
    };
    // Simply do nothing on most architectures.
    idxs
}

/// As transize but no implicit call to `zeroing_idxs`.
#[allow(dead_code)]
#[inline(always)]
unsafe fn transize_raw<T, const N: usize>(
    f: unsafe fn(T, T) -> T,
    bytes: Simd<u8, N>,
    idxs: Simd<u8, N>,
) -> Simd<u8, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    // SAFETY: Same obligation to use this function as to use mem::transmute_copy.
    unsafe { mem::transmute_copy(&f(mem::transmute_copy(&bytes), mem::transmute_copy(&idxs))) }
}
