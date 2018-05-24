//! `coresimd`

#[macro_use]
mod macros;

#[macro_use]
mod ppsv;

/// Platform independent SIMD vector types and operations.
///
/// This is an **unstable** module for portable SIMD operations. This module
/// has not yet gone through an RFC and is likely to change, but feedback is
/// always welcome!
#[unstable(feature = "stdsimd", issue = "27731")]
pub mod simd {
    pub use coresimd::ppsv::*;
}

/// Platform dependent vendor intrinsics.
///
/// This documentation is for the version of this module in the `coresimd`
/// crate, but you probably want to use the [`stdsimd` crate][stdsimd] which
/// should have more complete documentation.
///
/// [stdsimd]: https://rust-lang-nursery.github.io/stdsimd/x86_64/stdsimd/arch/index.html
///
/// Also note that while this module may appear to contains the intrinsics for
/// only one platform it actually contains intrinsics for multiple platforms
/// compiled in conditionally. For other platforms of stdsimd see:
///
/// * [`x86`]
/// * [`x86_64`]
/// * [`arm`]
/// * [`aarch64`]
/// * [`mips`]
/// * [`mips64`]
/// * [`PowerPC`]
/// * [`PowerPC64`]
///
/// [`x86`]: https://rust-lang-nursery.github.io/stdsimd/x86/stdsimd/arch/index.html
/// [`x86_64`]: https://rust-lang-nursery.github.io/stdsimd/x86_64/stdsimd/arch/index.html
/// [`arm`]: https://rust-lang-nursery.github.io/stdsimd/arm/stdsimd/arch/index.html
/// [`aarch64`]: https://rust-lang-nursery.github.io/stdsimd/aarch64/stdsimd/arch/index.html
/// [`mips`]: https://rust-lang-nursery.github.io/stdsimd/mips/stdsimd/arch/index.html
/// [`mips64`]: https://rust-lang-nursery.github.io/stdsimd/mips64/stdsimd/arch/index.html
/// [`PowerPC`]: https://rust-lang-nursery.github.io/stdsimd/powerpc/stdsimd/arch/index.html
/// [`PowerPC64`]: https://rust-lang-nursery.github.io/stdsimd/powerpc64/stdsimd/arch/index.html
#[stable(feature = "simd_arch", since = "1.27.0")]
pub mod arch {
    /// Platform-specific intrinsics for the `x86` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "x86", dox))]
    #[doc(cfg(target_arch = "x86"))]
    #[stable(feature = "simd_x86", since = "1.27.0")]
    pub mod x86 {
        #[stable(feature = "simd_x86", since = "1.27.0")]
        pub use coresimd::x86::*;
    }

    /// Platform-specific intrinsics for the `x86_64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "x86_64", dox))]
    #[doc(cfg(target_arch = "x86_64"))]
    #[stable(feature = "simd_x86", since = "1.27.0")]
    pub mod x86_64 {
        #[stable(feature = "simd_x86", since = "1.27.0")]
        pub use coresimd::x86::*;
        #[stable(feature = "simd_x86", since = "1.27.0")]
        pub use coresimd::x86_64::*;
    }

    /// Platform-specific intrinsics for the `arm` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "arm", dox))]
    #[doc(cfg(target_arch = "arm"))]
    #[unstable(feature = "stdsimd", issue = "27731")]
    pub mod arm {
        pub use coresimd::arm::*;
    }

    /// Platform-specific intrinsics for the `aarch64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "aarch64", dox))]
    #[doc(cfg(target_arch = "aarch64"))]
    #[unstable(feature = "stdsimd", issue = "27731")]
    pub mod aarch64 {
        pub use coresimd::aarch64::*;
        pub use coresimd::arm::*;
    }

    /// Platform-specific intrinsics for the `wasm32` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(target_arch = "wasm32")]
    #[unstable(feature = "stdsimd", issue = "27731")]
    pub mod wasm32 {
        pub use coresimd::wasm32::*;
    }

    /// Platform-specific intrinsics for the `mips` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "mips", dox))]
    #[doc(cfg(target_arch = "mips"))]
    #[unstable(feature = "stdsimd", issue = "27731")]
    pub mod mips {
        pub use coresimd::mips::*;
    }

    /// Platform-specific intrinsics for the `mips64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "mips64", dox))]
    #[doc(cfg(target_arch = "mips64"))]
    #[unstable(feature = "stdsimd", issue = "27731")]
    pub mod mips64 {
        pub use coresimd::mips::*;
    }

    /// Platform-specific intrinsics for the `PowerPC` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "powerpc", dox))]
    #[doc(cfg(target_arch = "powerpc"))]
    #[unstable(feature = "stdsimd", issue = "27731")]
    pub mod powerpc {
        pub use coresimd::powerpc::*;
    }

    /// Platform-specific intrinsics for the `PowerPC64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(target_arch = "powerpc64")]
    #[cfg(any(target_arch = "powerpc64", dox))]
    #[doc(cfg(target_arch = "powerpc64"))]
    #[unstable(feature = "stdsimd", issue = "27731")]
    pub mod powerpc64 {
        pub use coresimd::powerpc64::*;
    }
}

mod simd_llvm;

#[cfg(any(target_arch = "x86", target_arch = "x86_64", dox))]
mod x86;
#[cfg(any(target_arch = "x86_64", dox))]
mod x86_64;

#[cfg(any(target_arch = "aarch64", dox))]
mod aarch64;
#[cfg(any(target_arch = "arm", target_arch = "aarch64", dox))]
mod arm;
#[cfg(target_arch = "wasm32")]
mod wasm32;

#[cfg(any(target_arch = "mips", target_arch = "mips64", dox))]
mod mips;

#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64", dox))]
mod powerpc;

#[cfg(any(target_arch = "powerpc64", dox))]
mod powerpc64;

mod nvptx;
