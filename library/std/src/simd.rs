#[unstable(feature = "portable_simd", issue = "86656")]
pub use core::simd::*;

// "platform intrinsics" are essentially "codegen intrinsics"
extern "platform-intrinsic" {
    // ceil
    fn simd_ceil<T>(x: T) -> T;

    // floor
    fn simd_floor<T>(x: T) -> T;

    // round
    fn simd_round<T>(x: T) -> T;

    // trunc
    fn simd_trunc<T>(x: T) -> T;

    // fsqrt
    fn simd_fsqrt<T>(x: T) -> T;

    // fma
    fn simd_fma<T>(x: T, y: T, z: T) -> T;
}

// std can break a little coherence, as a treat!
// `f{32,64}{,_simd}_runtime` lang items let std impl these functions, breaking coherence.
// This is because Rust may require runtime support for these functions for some targets,
// as LLVM may "libcall legalize" a function call to using libm (aka math.h).
// To remove `f{32,64}_simd_runtime`, "simply" write a libmvec (math.h for SIMD) in Rust,
// and tell LLVM to compile it in if it needs to do those libcalls.
#[cfg(not(test))]
#[lang_item = "f32_simd_runtime"]
impl<const N: usize> Simd<f32, N> {
    /// Fused multiply-add.  Computes `(self * a) + b` with only one rounding error,
    /// yielding a more accurate result than an unfused multiply-add.
    ///
    /// Using `mul_add` *may* be more performant than an unfused multiply-add if the target
    /// architecture has a dedicated `fma` CPU instruction.  However, this is not always
    /// true, and will be heavily dependent on designing algorithms with specific target
    /// hardware in mind.
    #[cfg(feature = "std")]
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { simd_fma(self, a, b) }
    }

    /// Produces a vector where every lane has the square root value
    /// of the equivalently-indexed lane in `self`
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[cfg(feature = "std")]
    pub fn sqrt(self) -> Self {
        unsafe { simd_fsqrt(self) }
    }

    /// Returns the smallest integer greater than or equal to each lane.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    pub fn ceil(self) -> Self {
        unsafe { simd_ceil(self) }
    }

    /// Returns the largest integer value less than or equal to each lane.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    pub fn floor(self) -> Self {
        unsafe { simd_floor(self) }
    }

    /// Rounds to the nearest integer value. Ties round toward zero.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    pub fn round(self) -> Self {
        unsafe { simd_round(self) }
    }

    /// Returns the floating point's integer value, with its fractional part removed.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    pub fn trunc(self) -> Self {
        unsafe { simd_trunc(self) }
    }

    /// Returns the floating point's fractional value, with its integer part removed.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    pub fn fract(self) -> Self {
        self - self.trunc()
    }
}

// std can break a little coherence, as a treat!
// See note for f32_simd_runtime if you want to know why.
#[cfg(not(test))]
#[lang_item = "f64_simd_runtime"]
impl<const N: usize> Simd<f64, N> {
    /// Fused multiply-add.  Computes `(self * a) + b` with only one rounding error,
    /// yielding a more accurate result than an unfused multiply-add.
    ///
    /// Using `mul_add` *may* be more performant than an unfused multiply-add if the target
    /// architecture has a dedicated `fma` CPU instruction.  However, this is not always
    /// true, and will be heavily dependent on designing algorithms with specific target
    /// hardware in mind.
    #[cfg(feature = "std")]
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { simd_fma(self, a, b) }
    }

    /// Produces a vector where every lane has the square root value
    /// of the equivalently-indexed lane in `self`
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[cfg(feature = "std")]
    pub fn sqrt(self) -> Self {
        unsafe { simd_fsqrt(self) }
    }

    /// Returns the smallest integer greater than or equal to each lane.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    pub fn ceil(self) -> Self {
        unsafe { simd_ceil(self) }
    }

    /// Returns the largest integer value less than or equal to each lane.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    pub fn floor(self) -> Self {
        unsafe { simd_floor(self) }
    }

    /// Rounds to the nearest integer value. Ties round toward zero.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    pub fn round(self) -> Self {
        unsafe { simd_round(self) }
    }

    /// Returns the floating point's integer value, with its fractional part removed.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    pub fn trunc(self) -> Self {
        unsafe { simd_trunc(self) }
    }

    /// Returns the floating point's fractional value, with its integer part removed.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    pub fn fract(self) -> Self {
        self - self.trunc()
    }
}
