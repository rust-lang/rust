//! `core_arch`

#![allow(unknown_lints, unnecessary_transmutes)]

#[macro_use]
mod macros;

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64", doc))]
mod riscv_shared;

#[cfg(any(
    target_arch = "arm",
    target_arch = "aarch64",
    target_arch = "arm64ec",
    doc
))]
mod arm_shared;

#[cfg(any(target_arch = "loongarch32", target_arch = "loongarch64", doc))]
mod loongarch_shared;

mod simd;

#[doc = include_str!("core_arch_docs.md")]
#[stable(feature = "simd_arch", since = "1.27.0")]
pub mod arch {
    /// Platform-specific intrinsics for the `x86` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "x86", doc))]
    #[doc(cfg(target_arch = "x86"))]
    #[stable(feature = "simd_x86", since = "1.27.0")]
    pub mod x86 {
        #[stable(feature = "simd_x86", since = "1.27.0")]
        pub use crate::core_arch::x86::*;
    }

    /// Platform-specific intrinsics for the `x86_64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "x86_64", doc))]
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
    #[cfg(any(target_arch = "arm", doc))]
    #[doc(cfg(target_arch = "arm"))]
    #[unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")]
    pub mod arm {
        #[unstable(feature = "stdarch_arm_neon_intrinsics", issue = "111800")]
        pub use crate::core_arch::arm::*;
    }

    /// Platform-specific intrinsics for the `aarch64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec", doc))]
    #[doc(cfg(any(target_arch = "aarch64", target_arch = "arm64ec")))]
    #[stable(feature = "neon_intrinsics", since = "1.59.0")]
    pub mod aarch64 {
        #[stable(feature = "neon_intrinsics", since = "1.59.0")]
        pub use crate::core_arch::aarch64::*;
    }

    /// Platform-specific intrinsics for the `riscv32` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "riscv32", doc))]
    #[doc(cfg(any(target_arch = "riscv32")))]
    #[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
    pub mod riscv32 {
        pub use crate::core_arch::riscv_shared::*;
        pub use crate::core_arch::riscv32::*;
    }

    /// Platform-specific intrinsics for the `riscv64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "riscv64", doc))]
    #[doc(cfg(any(target_arch = "riscv64")))]
    #[unstable(feature = "riscv_ext_intrinsics", issue = "114544")]
    pub mod riscv64 {
        pub use crate::core_arch::riscv64::*;
        // RISC-V RV64 supports all RV32 instructions as well in current specifications (2022-01-05).
        // Module `riscv_shared` includes instructions available under all RISC-V platforms,
        // i.e. RISC-V RV32 instructions.
        pub use crate::core_arch::riscv_shared::*;
    }

    /// Platform-specific intrinsics for the `wasm32` platform.
    ///
    /// This module provides intrinsics specific to the WebAssembly
    /// architecture. Here you'll find intrinsics specific to WebAssembly that
    /// aren't otherwise surfaced somewhere in a cross-platform abstraction of
    /// `std`, and you'll also find functions for leveraging WebAssembly
    /// proposals such as [atomics] and [simd].
    ///
    /// Intrinsics in the `wasm32` module are modeled after the WebAssembly
    /// instructions that they represent. Most functions are named after the
    /// instruction they intend to correspond to, and the arguments/results
    /// correspond to the type signature of the instruction itself. Stable
    /// WebAssembly instructions are [documented online][instrdoc].
    ///
    /// [instrdoc]: https://webassembly.github.io/spec/core/valid/instructions.html
    ///
    /// If a proposal is not yet stable in WebAssembly itself then the functions
    /// within this function may be unstable and require the nightly channel of
    /// Rust to use. As the proposal itself stabilizes the intrinsics in this
    /// module should stabilize as well.
    ///
    /// [atomics]: https://github.com/webassembly/threads
    /// [simd]: https://github.com/webassembly/simd
    ///
    /// See the [module documentation](../index.html) for general information
    /// about the `arch` module and platform intrinsics.
    ///
    /// ## Atomics
    ///
    /// The [threads proposal][atomics] for WebAssembly adds a number of
    /// instructions for dealing with multithreaded programs. Most instructions
    /// added in the [atomics] proposal are exposed in Rust through the
    /// `std::sync::atomic` module. Some instructions, however, don't have
    /// direct equivalents in Rust so they're exposed here instead.
    ///
    /// Note that the instructions added in the [atomics] proposal can work in
    /// either a context with a shared wasm memory and without. These intrinsics
    /// are always available in the standard library, but you likely won't be
    /// able to use them too productively unless you recompile the standard
    /// library (and all your code) with `-Ctarget-feature=+atomics`.
    ///
    /// It's also worth pointing out that multi-threaded WebAssembly and its
    /// story in Rust is still in a somewhat "early days" phase as of the time
    /// of this writing. Pieces should mostly work but it generally requires a
    /// good deal of manual setup. At this time it's not as simple as "just call
    /// `std::thread::spawn`", but it will hopefully get there one day!
    ///
    /// ## SIMD
    ///
    /// The [simd proposal][simd] for WebAssembly added a new `v128` type for a
    /// 128-bit SIMD register. It also added a large array of instructions to
    /// operate on the `v128` type to perform data processing. Using SIMD on
    /// wasm is intended to be similar to as you would on `x86_64`, for example.
    /// You'd write a function such as:
    ///
    /// ```rust,ignore
    /// #[cfg(target_arch = "wasm32")]
    /// #[target_feature(enable = "simd128")]
    /// unsafe fn uses_simd() {
    ///     use std::arch::wasm32::*;
    ///     // ...
    /// }
    /// ```
    ///
    /// Unlike `x86_64`, however, WebAssembly does not currently have dynamic
    /// detection at runtime as to whether SIMD is supported (this is one of the
    /// motivators for the [conditional sections][condsections] and [feature
    /// detection] proposals, but that is still pretty early days). This means
    /// that your binary will either have SIMD and can only run on engines
    /// which support SIMD, or it will not have SIMD at all. For compatibility
    /// the standard library itself does not use any SIMD internally.
    /// Determining how best to ship your WebAssembly binary with SIMD is
    /// largely left up to you as it can be pretty nuanced depending on
    /// your situation.
    ///
    /// [condsections]: https://github.com/webassembly/conditional-sections
    /// [feature detection]: https://github.com/WebAssembly/feature-detection
    ///
    /// To enable SIMD support at compile time you need to do one of two things:
    ///
    /// * First you can annotate functions with `#[target_feature(enable =
    ///   "simd128")]`. This causes just that one function to have SIMD support
    ///   available to it, and intrinsics will get inlined as usual in this
    ///   situation.
    ///
    /// * Second you can compile your program with `-Ctarget-feature=+simd128`.
    ///   This compilation flag blanket enables SIMD support for your entire
    ///   compilation. Note that this does not include the standard library
    ///   unless you [recompile the standard library][buildstd].
    ///
    /// [buildstd]: https://doc.rust-lang.org/nightly/cargo/reference/unstable.html#build-std
    ///
    /// If you enable SIMD via either of these routes then you'll have a
    /// WebAssembly binary that uses SIMD instructions, and you'll need to ship
    /// that accordingly. Also note that if you call SIMD intrinsics but don't
    /// enable SIMD via either of these mechanisms, you'll still have SIMD
    /// generated in your program. This means to generate a binary without SIMD
    /// you'll need to avoid both options above plus calling into any intrinsics
    /// in this module.
    #[cfg(any(target_arch = "wasm32", doc))]
    #[doc(cfg(target_arch = "wasm32"))]
    #[stable(feature = "simd_wasm32", since = "1.33.0")]
    pub mod wasm32 {
        #[stable(feature = "simd_wasm32", since = "1.33.0")]
        pub use crate::core_arch::wasm32::*;
    }

    /// Platform-specific intrinsics for the `wasm64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "wasm64", doc))]
    #[doc(cfg(target_arch = "wasm64"))]
    #[unstable(feature = "simd_wasm64", issue = "90599")]
    pub mod wasm64 {
        #[unstable(feature = "simd_wasm64", issue = "90599")]
        pub use crate::core_arch::wasm32::*;
    }

    /// Platform-specific intrinsics for the `wasm` target family.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_family = "wasm", doc))]
    #[doc(cfg(target_family = "wasm"))]
    #[unstable(feature = "simd_wasm64", issue = "90599")]
    pub mod wasm {
        #[unstable(feature = "simd_wasm64", issue = "90599")]
        pub use crate::core_arch::wasm32::*;
    }

    /// Platform-specific intrinsics for the `mips` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "mips", doc))]
    #[doc(cfg(target_arch = "mips"))]
    #[unstable(feature = "stdarch_mips", issue = "111198")]
    pub mod mips {
        pub use crate::core_arch::mips::*;
    }

    /// Platform-specific intrinsics for the `mips64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "mips64", doc))]
    #[doc(cfg(target_arch = "mips64"))]
    #[unstable(feature = "stdarch_mips", issue = "111198")]
    pub mod mips64 {
        pub use crate::core_arch::mips::*;
    }

    /// Platform-specific intrinsics for the `PowerPC` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "powerpc", doc))]
    #[doc(cfg(target_arch = "powerpc"))]
    #[unstable(feature = "stdarch_powerpc", issue = "111145")]
    pub mod powerpc {
        pub use crate::core_arch::powerpc::*;
    }

    /// Platform-specific intrinsics for the `PowerPC64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "powerpc64", doc))]
    #[doc(cfg(target_arch = "powerpc64"))]
    #[unstable(feature = "stdarch_powerpc", issue = "111145")]
    pub mod powerpc64 {
        pub use crate::core_arch::powerpc64::*;
    }

    /// Platform-specific intrinsics for the `NVPTX` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "nvptx64", doc))]
    #[doc(cfg(target_arch = "nvptx64"))]
    #[unstable(feature = "stdarch_nvptx", issue = "111199")]
    pub mod nvptx {
        pub use crate::core_arch::nvptx::*;
    }

    /// Platform-specific intrinsics for the `loongarch32` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "loongarch32", doc))]
    #[doc(cfg(target_arch = "loongarch32"))]
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub mod loongarch32 {
        pub use crate::core_arch::loongarch_shared::*;
        pub use crate::core_arch::loongarch32::*;
    }

    /// Platform-specific intrinsics for the `loongarch64` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "loongarch64", doc))]
    #[doc(cfg(target_arch = "loongarch64"))]
    #[unstable(feature = "stdarch_loongarch", issue = "117427")]
    pub mod loongarch64 {
        pub use crate::core_arch::loongarch_shared::*;
        pub use crate::core_arch::loongarch64::*;
    }

    /// Platform-specific intrinsics for the `s390x` platform.
    ///
    /// See the [module documentation](../index.html) for more details.
    #[cfg(any(target_arch = "s390x", doc))]
    #[doc(cfg(target_arch = "s390x"))]
    #[unstable(feature = "stdarch_s390x", issue = "135681")]
    pub mod s390x {
        pub use crate::core_arch::s390x::*;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64", doc))]
#[doc(cfg(any(target_arch = "x86", target_arch = "x86_64")))]
mod x86;
#[cfg(any(target_arch = "x86_64", doc))]
#[doc(cfg(target_arch = "x86_64"))]
mod x86_64;

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec", doc))]
#[doc(cfg(any(target_arch = "aarch64", target_arch = "arm64ec")))]
mod aarch64;
#[cfg(any(target_arch = "arm", doc))]
#[doc(cfg(any(target_arch = "arm")))]
mod arm;

#[cfg(any(target_arch = "riscv32", doc))]
#[doc(cfg(any(target_arch = "riscv32")))]
mod riscv32;

#[cfg(any(target_arch = "riscv64", doc))]
#[doc(cfg(any(target_arch = "riscv64")))]
mod riscv64;

#[cfg(any(target_family = "wasm", doc))]
#[doc(cfg(target_family = "wasm"))]
mod wasm32;

#[cfg(any(target_arch = "mips", target_arch = "mips64", doc))]
#[doc(cfg(any(target_arch = "mips", target_arch = "mips64")))]
mod mips;

#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64", doc))]
#[doc(cfg(any(target_arch = "powerpc", target_arch = "powerpc64")))]
mod powerpc;

#[cfg(any(target_arch = "powerpc64", doc))]
#[doc(cfg(target_arch = "powerpc64"))]
mod powerpc64;

#[cfg(any(target_arch = "nvptx64", doc))]
#[doc(cfg(target_arch = "nvptx64"))]
mod nvptx;

#[cfg(any(target_arch = "loongarch32", doc))]
#[doc(cfg(target_arch = "loongarch32"))]
mod loongarch32;

#[cfg(any(target_arch = "loongarch64", doc))]
#[doc(cfg(target_arch = "loongarch64"))]
mod loongarch64;

#[cfg(any(target_arch = "s390x", doc))]
#[doc(cfg(target_arch = "s390x"))]
mod s390x;
