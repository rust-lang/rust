//! `core_arch`

#[macro_use]
mod macros;

#[cfg(any(target_arch = "arm", target_arch = "aarch64", dox))]
mod acle;

mod simd;

#[doc(include = "core_arch_docs.md")]
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
        pub use crate::core_arch::x86::*;
    }

    /// Platform-specific intrinsics for the `x86_64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "x86_64", dox))]
    #[doc(cfg(target_arch = "x86_64"))]
    #[stable(feature = "simd_x86", since = "1.27.0")]
    pub mod x86_64 {
        #[stable(feature = "simd_x86", since = "1.27.0")]
        pub use crate::core_arch::x86::*;
        #[stable(feature = "simd_x86", since = "1.27.0")]
        pub use crate::core_arch::x86_64::*;
    }

    /// Platform-specific intrinsics for the `arm` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "arm", dox))]
    #[doc(cfg(target_arch = "arm"))]
    #[unstable(feature = "stdsimd", issue = "27731")]
    pub mod arm {
        pub use crate::core_arch::arm::*;
    }

    /// Platform-specific intrinsics for the `aarch64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "aarch64", dox))]
    #[doc(cfg(target_arch = "aarch64"))]
    #[unstable(feature = "stdsimd", issue = "27731")]
    pub mod aarch64 {
        pub use crate::core_arch::{aarch64::*, arm::*};
    }

    /// Platform-specific intrinsics for the `wasm32` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "wasm32", dox))]
    #[doc(cfg(target_arch = "wasm32"))]
    #[stable(feature = "simd_wasm32", since = "1.33.0")]
    pub mod wasm32 {
        #[stable(feature = "simd_wasm32", since = "1.33.0")]
        pub use crate::core_arch::wasm32::*;
    }

    /// Platform-specific intrinsics for the `mips` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "mips", dox))]
    #[doc(cfg(target_arch = "mips"))]
    #[unstable(feature = "stdsimd", issue = "27731")]
    pub mod mips {
        pub use crate::core_arch::mips::*;
    }

    /// Platform-specific intrinsics for the `mips64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "mips64", dox))]
    #[doc(cfg(target_arch = "mips64"))]
    #[unstable(feature = "stdsimd", issue = "27731")]
    pub mod mips64 {
        pub use crate::core_arch::mips::*;
    }

    /// Platform-specific intrinsics for the `PowerPC` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "powerpc", dox))]
    #[doc(cfg(target_arch = "powerpc"))]
    #[unstable(feature = "stdsimd", issue = "27731")]
    pub mod powerpc {
        pub use crate::core_arch::powerpc::*;
    }

    /// Platform-specific intrinsics for the `PowerPC64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "powerpc64", dox))]
    #[doc(cfg(target_arch = "powerpc64"))]
    #[unstable(feature = "stdsimd", issue = "27731")]
    pub mod powerpc64 {
        pub use crate::core_arch::powerpc64::*;
    }

    /// Platform-specific intrinsics for the `NVPTX` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "nvptx", target_arch = "nvptx64", dox))]
    #[doc(cfg(any(target_arch = "nvptx", target_arch = "nvptx64")))]
    #[unstable(feature = "stdsimd", issue = "27731")]
    pub mod nvptx {
        pub use crate::core_arch::nvptx::*;
    }
}

mod simd_llvm;

#[cfg(any(target_arch = "x86", target_arch = "x86_64", dox))]
#[doc(cfg(any(target_arch = "x86", target_arch = "x86_64")))]
mod x86;
#[cfg(any(target_arch = "x86_64", dox))]
#[doc(cfg(target_arch = "x86_64"))]
mod x86_64;

#[cfg(any(target_arch = "aarch64", dox))]
#[doc(cfg(target_arch = "aarch64"))]
mod aarch64;
#[cfg(any(target_arch = "arm", target_arch = "aarch64", dox))]
#[doc(cfg(any(target_arch = "arm", target_arch = "aarch64")))]
mod arm;

#[cfg(any(target_arch = "wasm32", dox))]
#[doc(cfg(target_arch = "wasm32"))]
mod wasm32;

#[cfg(any(target_arch = "mips", target_arch = "mips64", dox))]
#[doc(cfg(any(target_arch = "mips", target_arch = "mips64")))]
mod mips;

#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64", dox))]
#[doc(cfg(any(target_arch = "powerpc", target_arch = "powerpc64")))]
mod powerpc;

#[cfg(any(target_arch = "powerpc64", dox))]
#[doc(cfg(target_arch = "powerpc64"))]
mod powerpc64;

#[cfg(any(target_arch = "nvptx", target_arch = "nvptx64", dox))]
#[doc(cfg(any(target_arch = "nvptx", target_arch = "nvptx64")))]
mod nvptx;
