//! LLVM6 currently generates sub-optimal code for the `all` mask reductions.
//!
//! See https://github.com/rust-lang-nursery/stdsimd/issues/362#issuecomment-372774371
//! and the associated LLVM bug:
//! https://bugs.llvm.org/show_bug.cgi?id=36702

#![allow(unused)]

use coresimd::simd::*;

pub trait All: ::marker::Sized {
    unsafe fn all(self) -> bool;
}

pub trait Any: ::marker::Sized {
    unsafe fn any(self) -> bool;
}

// By default we use the simd_reduce_{all,any} intrinsics, which produces
// sub-optimal code, except on aarch64 where that intrinsic is broken
// due to https://bugs.llvm.org/show_bug.cgi?id=36796 so we just use
// full-blown bitwise and/or reduction there.
macro_rules! default_impl {
    ($id:ident) => {
        impl All for $id {
            #[inline]
            unsafe fn all(self) -> bool {
                #[cfg(not(target_arch = "aarch64"))] {
                    use coresimd::simd_llvm::simd_reduce_all;
                    simd_reduce_all(self)
                }
                #[cfg(target_arch = "aarch64")] {
                    // FIXME: Broken on AArch64
                    // https://bugs.llvm.org/show_bug.cgi?id=36796
                    self.and()
                }
            }
        }

        impl Any for $id {
            #[inline]
            unsafe fn any(self) -> bool {
                #[cfg(not(target_arch = "aarch64"))] {
                    use coresimd::simd_llvm::simd_reduce_any;
                    simd_reduce_any(self)
                }
                #[cfg(target_arch = "aarch64")] {
                    // FIXME: Broken on AArch64
                    // https://bugs.llvm.org/show_bug.cgi?id=36796
                    self.or()
                }
            }
        }
    };
}

// On x86 both SSE2 and AVX2 provide movemask instructions that can be used
// here. The AVX2 instructions aren't necessarily better than the AVX
// instructions below, so they aren't implemented here.
//
// FIXME: for mask generated from f32x4 LLVM6 emits pmovmskb but should emit
// movmskps. Since the masks don't track whether they were produced by integer
// or floating point vectors, we can't currently work around this yet. The
// performance impact for this shouldn't be large, but this is filled as:
// https://bugs.llvm.org/show_bug.cgi?id=37087
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"))]
macro_rules! x86_128_sse2_movemask_impl {
    ($id:ident) => {
        impl All for $id {
            #[inline]
            #[target_feature(enable = "sse2")]
            unsafe fn all(self) -> bool {
                #[cfg(target_arch = "x86")]
                use ::coresimd::arch::x86::_mm_movemask_epi8;
                #[cfg(target_arch = "x86_64")]
                use ::coresimd::arch::x86_64::_mm_movemask_epi8;
                // _mm_movemask_epi8(a) creates a 16bit mask containing the most
                // significant bit of each byte of `a`. If all bits are set,
                // then all 16 lanes of the mask are true.
                _mm_movemask_epi8(::mem::transmute(self)) == u16::max_value() as i32
            }
        }
        impl Any for $id {
            #[inline]
            #[target_feature(enable = "sse2")]
            unsafe fn any(self) -> bool {
                #[cfg(target_arch = "x86")]
                use ::coresimd::arch::x86::_mm_movemask_epi8;
                #[cfg(target_arch = "x86_64")]
                use ::coresimd::arch::x86_64::_mm_movemask_epi8;

                _mm_movemask_epi8(::mem::transmute(self)) != 0
            }
        }
    }
}

// On x86 with AVX we use _mm256_testc_si256 and _mm256_testz_si256.
//
// FIXME: for masks generated from floating point vectors one should use
// x86_mm256_testc_ps, x86_mm256_testz_ps, x86_mm256_testc_pd,
// x86_mm256_testz_pd.Since the masks don't track whether they were produced by
// integer or floating point vectors, we can't currently work around this yet.
//
// TODO: investigate perf impact and fill LLVM bugs as necessary.
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx"))]
macro_rules! x86_256_avx_test_impl {
    ($id:ident) => {
        impl All for $id {
            #[inline]
            #[target_feature(enable = "avx")]
            unsafe fn all(self) -> bool {
                #[cfg(target_arch = "x86")]
                use ::coresimd::arch::x86::_mm256_testc_si256;
                #[cfg(target_arch = "x86_64")]
                use ::coresimd::arch::x86_64::_mm256_testc_si256;
                _mm256_testc_si256(::mem::transmute(self),
                                   ::mem::transmute($id::splat(true))) != 0
            }
        }
        impl Any for $id {
            #[inline]
            #[target_feature(enable = "avx")]
            unsafe fn any(self) -> bool {
                #[cfg(target_arch = "x86")]
                use ::coresimd::arch::x86::_mm256_testz_si256;
                #[cfg(target_arch = "x86_64")]
                use ::coresimd::arch::x86_64::_mm256_testz_si256;
                _mm256_testz_si256(::mem::transmute(self),
                                   ::mem::transmute(self)) == 0
            }
        }
    }
}

// On x86 with SSE2 all/any for 256-bit wide vectors is implemented by executing
// the algorithm for 128-bit on the higher and lower elements of the vector
// independently.
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2"))]
macro_rules! x86_256_sse2_impl {
    ($id:ident, $v128:ident) => {
        impl All for $id {
            #[inline]
            #[target_feature(enable = "sse2")]
            unsafe fn all(self) -> bool {
                unsafe {
                    union U {
                        halves: ($v128, $v128),
                        vec: $id
                    }
                    let halves = U {vec: self}.halves;
                    halves.0.all() && halves.1.all()
                }
            }
        }
        impl Any for $id {
            #[inline]
            #[target_feature(enable = "sse2")]
            unsafe fn any(self) -> bool {
                unsafe {
                    union U {
                        halves: ($v128, $v128),
                        vec: $id
                    }
                    let halves = U {vec: self}.halves;
                    halves.0.any() || halves.1.any()
                }
            }
        }
    }
}

// Implementation for 64-bit wide masks on x86.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
macro_rules! x86_64_mmx_movemask_impl {
    ($id:ident, $vec128:ident) => {
        impl All for $id {
            #[inline]
            #[target_feature(enable = "mmx")]
            unsafe fn all(self) -> bool {
                #[cfg(target_arch = "x86")]
                use ::coresimd::arch::x86::_mm_movemask_pi8;
                #[cfg(target_arch = "x86_64")]
                use ::coresimd::arch::x86_64::_mm_movemask_pi8;
                // _mm_movemask_pi8(a) creates an 8bit mask containing the most
                // significant bit of each byte of `a`. If all bits are set,
                // then all 8 lanes of the mask are true.
                 _mm_movemask_pi8(::mem::transmute(self)) == u8::max_value() as i32
            }
        }
        impl Any for $id {
            #[inline]
            #[target_feature(enable = "mmx")]
            unsafe fn any(self) -> bool {
                #[cfg(target_arch = "x86")]
                use ::coresimd::arch::x86::_mm_movemask_pi8;
                #[cfg(target_arch = "x86_64")]
                use ::coresimd::arch::x86_64::_mm_movemask_pi8;

                _mm_movemask_pi8(::mem::transmute(self)) != 0
            }
        }
    }
}

// Implementation for 128-bit wide masks on x86
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
macro_rules! x86_128_impl {
    ($id:ident) => {
        cfg_if! {
            if #[cfg(target_feature = "sse2")] {
                x86_128_sse2_movemask_impl!($id);
            }  else {
                default_impl!($id);
            }
        }
    }
}

// Implementation for 256-bit wide masks on x86
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
macro_rules! x86_256_impl {
    ($id:ident, $half_id:ident) => {
        cfg_if! {
            if #[cfg(target_feature = "avx")] {
                x86_256_avx_test_impl!($id);
            } else if #[cfg(target_feature = "sse2")] {
                x86_256_sse2_impl!($id, $half_id);
            } else {
                default_impl!($id);
            }
        }
    }
}

// Implementation for ARM + v7 + NEON using vpmin and vpmax (folding
// minimum/maximum of adjacent pairs) for 64-bit wide two-element vectors.
#[cfg(all(target_arch = "arm", target_feature = "v7", target_feature = "neon"))]
macro_rules! arm_64_x2_v7_neon_impl {
    ($id:ident, $vpmin:ident, $vpmax:ident) => {
        impl All for $id {
            #[inline]
            #[target_feature(enable = "v7,neon")]
            unsafe fn all(self) -> bool {
                use ::coresimd::arch::arm::$vpmin;
                use ::mem::transmute;
                // pmin((a, b), (-,-)) => (b, -).0 => b
                let tmp: $id = transmute($vpmin(transmute(self), ::mem::uninitialized()));
                tmp.extract(0)
            }
        }
        impl Any for $id {
            #[inline]
            #[target_feature(enable = "v7,neon")]
            unsafe fn any(self) -> bool {
                use ::coresimd::arch::arm::$vpmax;
                use ::mem::transmute;
                // pmax((a, b), (-,-)) => (b, -).0 => b
                let tmp: $id = transmute($vpmax(transmute(self), ::mem::uninitialized()));
                tmp.extract(0)
            }
        }
    }
}

// Implementation for ARM + v7 + NEON using vpmin and vpmax (folding
// minimum/maximum of adjacent pairs) for 64-bit wide four-element vectors.
#[cfg(all(target_arch = "arm", target_feature = "v7", target_feature = "neon"))]
macro_rules! arm_64_x4_v7_neon_impl {
    ($id:ident, $vpmin:ident, $vpmax:ident) => {
        impl All for $id {
            #[inline]
            #[target_feature(enable = "v7,neon")]
            unsafe fn all(self) -> bool {
                use ::coresimd::arch::arm::$vpmin;
                use ::mem::transmute;
                // tmp = pmin((a, b, c, d), (-,-,-,-)) => (a, c, -, -)
                let tmp = $vpmin(transmute(self), ::mem::uninitialized());
                // tmp = pmin((a, b, -, -), (-,-,-,-)) => (c, -, -, -).0 => c
                let tmp: $id = transmute($vpmin(tmp, ::mem::uninitialized()));
                tmp.extract(0)
            }
        }
        impl Any for $id {
            #[inline]
            #[target_feature(enable = "v7,neon")]
            unsafe fn any(self) -> bool {
                use ::coresimd::arch::arm::$vpmax;
                use ::mem::transmute;
                // tmp = pmax((a, b, c, d), (-,-,-,-)) => (a, c, -, -)
                let tmp =  $vpmax(transmute(self), ::mem::uninitialized());
                // tmp = pmax((a, b, -, -), (-,-,-,-)) => (c, -, -, -).0 => c
                let tmp: $id = transmute($vpmax(tmp, ::mem::uninitialized()));
                tmp.extract(0)
            }
        }
    }
}

// Implementation for ARM + v7 + NEON using vpmin and vpmax (folding
// minimum/maximum of adjacent pairs) for 64-bit wide eight-element vectors.
#[cfg(all(target_arch = "arm", target_feature = "v7", target_feature = "neon"))]
macro_rules! arm_64_x8_v7_neon_impl {
    ($id:ident, $vpmin:ident, $vpmax:ident) => {
        impl All for $id {
            #[inline]
            #[target_feature(enable = "v7,neon")]
            unsafe fn all(self) -> bool {
                use ::coresimd::arch::arm::$vpmin;
                use ::mem::transmute;
                // tmp = pmin(
                //     (a, b, c, d, e, f, g, h),
                //     (-, -, -, -, -, -, -, -)
                // ) => (a, c, e, g, -, -, -, -)
                let tmp = $vpmin(transmute(self), ::mem::uninitialized());
                // tmp = pmin(
                //     (a, c, e, g, -, -, -, -),
                //     (-, -, -, -, -, -, -, -)
                // ) => (c, g, -, -, -, -, -, -)
                let tmp = $vpmin(tmp, ::mem::uninitialized());
                // tmp = pmin(
                //     (c, g, -, -, -, -, -, -),
                //     (-, -, -, -, -, -, -, -)
                // ) => (g, -, -, -, -, -, -, -).0 => g
                let tmp: $id = transmute($vpmin(tmp, ::mem::uninitialized()));
                tmp.extract(0)
            }
        }
        impl Any for $id {
            #[inline]
            #[target_feature(enable = "v7,neon")]
            unsafe fn any(self) -> bool {
                use ::coresimd::arch::arm::$vpmax;
                use ::mem::transmute;
                // tmp = pmax(
                //     (a, b, c, d, e, f, g, h),
                //     (-, -, -, -, -, -, -, -)
                // ) => (a, c, e, g, -, -, -, -)
                let tmp = $vpmax(transmute(self), ::mem::uninitialized());
                // tmp = pmax(
                //     (a, c, e, g, -, -, -, -),
                //     (-, -, -, -, -, -, -, -)
                // ) => (c, g, -, -, -, -, -, -)
                let tmp = $vpmax(tmp, ::mem::uninitialized());
                // tmp = pmax(
                //     (c, g, -, -, -, -, -, -),
                //     (-, -, -, -, -, -, -, -)
                // ) => (g, -, -, -, -, -, -, -).0 => g
                let tmp: $id = transmute($vpmax(tmp, ::mem::uninitialized()));
                tmp.extract(0)
            }
        }
    }
}


// Implementation for ARM + v7 + NEON using vpmin and vpmax (folding
// minimum/maximum of adjacent pairs) for 64-bit or 128-bit wide vectors with
// more than two elements.
#[cfg(all(target_arch = "arm", target_feature = "v7", target_feature = "neon"))]
macro_rules! arm_128_v7_neon_impl {
    ($id:ident, $half:ident, $vpmin:ident, $vpmax:ident) => {
        impl All for $id {
            #[inline]
            #[target_feature(enable = "v7,neon")]
            unsafe fn all(self) -> bool {
                use ::coresimd::arch::arm::$vpmin;
                use ::mem::transmute;
                union U {
                    halves: ($half, $half),
                    vec: $id
                }
                let halves = U { vec: self }.halves;
                let h: $half = transmute($vpmin(transmute(halves.0), transmute(halves.1)));
                h.all()
            }
        }
        impl Any for $id {
            #[inline]
            #[target_feature(enable = "v7,neon")]
            unsafe fn any(self) -> bool {
                use ::coresimd::arch::arm::$vpmax;
                use ::mem::transmute;
                union U {
                    halves: ($half, $half),
                    vec: $id
                }
                let halves = U { vec: self }.halves;
                let h: $half = transmute($vpmax(transmute(halves.0), transmute(halves.1)));
                h.any()
            }
        }
    }
}

// Implementation for AArch64 + NEON using vmin and vmax (horizontal vector
// min/max) for 128-bit wide vectors.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
macro_rules! aarch64_128_neon_impl {
    ($id:ident, $vmin:ident, $vmax:ident) => {
        impl All for $id {
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn all(self) -> bool {
                use ::coresimd::arch::aarch64::$vmin;
                $vmin(::mem::transmute(self)) != 0
            }
        }
        impl Any for $id {
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn any(self) -> bool {
                use ::coresimd::arch::aarch64::$vmax;
                $vmax(::mem::transmute(self)) != 0
            }
        }
    }
}

// Implementation for AArch64 + NEON using vmin and vmax (horizontal vector
// min/max) for 64-bit wide vectors.
//
// This impl duplicates the 64-bit vector into a 128-bit one and calls
// all/any on that.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
macro_rules! aarch64_64_neon_impl {
    ($id:ident, $vec128:ident) => {
        impl All for $id {
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn all(self) -> bool {
                union U {
                    halves: ($id, $id),
                    vec: $vec128
                }
                U { halves: (self, self) }.vec.all()
            }
        }
        impl Any for $id {
            #[inline]
            #[target_feature(enable = "neon")]
            unsafe fn any(self) -> bool {
                union U {
                    halves: ($id, $id),
                    vec: $vec128
                }
                U { halves: (self, self) }.vec.any()
            }
        }
    }
}

macro_rules! impl_mask_all_any {
    // 64-bit wide masks
    (m8x8) => {
        cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                x86_64_mmx_movemask_impl!(m8x8, m8x16);
            } else if #[cfg(all(target_arch = "arm", target_feature = "v7", target_feature = "neon"))] {
                arm_64_x8_v7_neon_impl!(m8x8, vpmin_u8, vpmax_u8);
            } else if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
                aarch64_64_neon_impl!(m8x8, m8x16);
            } else {
                default_impl!(m8x8);
            }
        }
    };
    (m16x4) => {
        cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                x86_64_mmx_movemask_impl!(m16x4, m16x8);
            } else if #[cfg(all(target_arch = "arm", target_feature = "v7", target_feature = "neon"))] {
                arm_64_x4_v7_neon_impl!(m16x4, vpmin_u16, vpmax_u16);
            } else if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
                aarch64_64_neon_impl!(m16x4, m16x8);
            } else {
                default_impl!(m16x4);
            }
        }
    };
    (m32x2) => {
        cfg_if! {
            if #[cfg(all(target_arch = "x86_64", not(target_os = "macos")))] {
                // FIXME: this fails on travis-ci osx build bots.
                x86_64_mmx_movemask_impl!(m32x2, m32x4);
            } else if #[cfg(all(target_arch = "arm", target_feature = "v7", target_feature = "neon"))] {
                arm_64_x2_v7_neon_impl!(m32x2, vpmin_u32, vpmax_u32);
            } else if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
                aarch64_64_neon_impl!(m32x2, m32x4);
            } else {
                default_impl!(m32x2);
            }
        }
    };
    // 128-bit wide masks
    (m8x16) => {
        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                x86_128_impl!(m8x16);
            } else if #[cfg(all(target_arch = "arm", target_feature = "v7", target_feature = "neon"))] {
                arm_128_v7_neon_impl!(m8x16, m8x8, vpmin_u8, vpmax_u8);
            } else if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
                aarch64_128_neon_impl!(m8x16, vminvq_u8, vmaxvq_u8);
            } else {
                default_impl!(m8x16);
            }
        }
    };
    (m16x8) => {
        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                x86_128_impl!(m16x8);
            } else if #[cfg(all(target_arch = "arm", target_feature = "v7", target_feature = "neon"))] {
                arm_128_v7_neon_impl!(m16x8, m16x4, vpmin_u16, vpmax_u16);
            } else if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
                aarch64_128_neon_impl!(m16x8, vminvq_u16, vmaxvq_u16);
            } else {
                default_impl!(m16x8);
            }
        }
    };
    (m32x4) => {
        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                x86_128_impl!(m32x4);
            } else if #[cfg(all(target_arch = "arm", target_feature = "v7", target_feature = "neon"))] {
                arm_128_v7_neon_impl!(m32x4, m32x2, vpmin_u32, vpmax_u32);
            } else if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
                aarch64_128_neon_impl!(m32x4, vminvq_u32, vmaxvq_u32);
            } else {
                default_impl!(m32x4);
            }
        }
    };
    (m64x2) => {
        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                x86_128_impl!(m64x2);
            } else {
                default_impl!(m64x2);
            }
        }
    };
    // 256-bit wide masks:
    (m8x32) => {
        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                x86_256_impl!(m8x32, m8x16);
            } else {
                default_impl!(m8x32);
            }
        }
    };
    (m16x16) => {
        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                x86_256_impl!(m16x16, m16x8);
            } else {
                default_impl!(m16x16);
            }
        }
    };
    (m32x8) => {
        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                x86_256_impl!(m32x8, m32x4);
            } else {
                default_impl!(m32x8);
            }
        }
    };
    (m64x4) => {
        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                x86_256_impl!(m64x4, m64x2);
            } else {
                default_impl!(m64x4);
            }
        }
    };
    // Fallback to LLVM's default code-generation:
    ($id:ident) => { default_impl!($id); };
}

vector_impl!(
    [impl_mask_all_any, m1x8],
    [impl_mask_all_any, m1x16],
    [impl_mask_all_any, m1x32],
    [impl_mask_all_any, m1x64],
    [impl_mask_all_any, m8x2],
    [impl_mask_all_any, m8x4],
    [impl_mask_all_any, m8x8],
    [impl_mask_all_any, m8x16],
    [impl_mask_all_any, m8x32],
    [impl_mask_all_any, m16x2],
    [impl_mask_all_any, m16x4],
    [impl_mask_all_any, m16x8],
    [impl_mask_all_any, m16x16],
    [impl_mask_all_any, m32x2],
    [impl_mask_all_any, m32x4],
    [impl_mask_all_any, m32x8],
    [impl_mask_all_any, m64x2],
    [impl_mask_all_any, m64x4]
);
