//! Architecture-specific routines and operations.
//!
//! LLVM will already optimize calls to some of these in cases that there are hardware
//! instructions. Providing an implementation here just ensures that the faster implementation
//! is used when calling the function directly. This helps anyone who uses `libm` directly, as
//! well as improving things when these routines are called as part of other implementations.

// Most implementations should be defined here, to ensure they are not made available when
// soft floats are required. Primarily gated on the `arch` feature, Miri does not support inline
// assembly.
#[cfg(all(feature = "arch", not(miri)))]
cfg_select_nofmt! {
    all(target_arch = "wasm32", intrinsics_enabled) => {
        mod wasm32;
        pub use wasm32::{
            ceil, ceilf, fabs, fabsf, floor, floorf, rint, rintf, sqrt, sqrtf, trunc, truncf,
        };
    }
    target_feature = "sse2" => {
        mod x86;
        pub use x86::{sqrt, sqrtf, fma, fmaf};
    }
    all(
        any(target_arch = "aarch64", target_arch = "arm64ec"),
        target_feature = "neon"
    ) => {
        mod aarch64;

        pub use aarch64::{
            fma,
            fmaf,
            rint,
            rintf,
            sqrt,
            sqrtf,
        };

        #[cfg(all(f16_enabled, target_feature = "fp16"))]
        pub use aarch64::{
            rintf16,
            sqrtf16,
        };
    }
    any(target_arch = "loongarch32", target_arch = "loongarch64") => {
        mod loongarch;

        #[cfg(target_feature = "d")]
        pub use loongarch::fma;
        #[cfg(target_feature = "f")]
        pub use loongarch::fmaf;
        #[cfg(target_feature = "lsx")]
        pub use loongarch::{rint, rintf};
        #[cfg(target_feature = "d")]
        pub use loongarch::sqrt;
        #[cfg(target_feature = "f")]
        pub use loongarch::sqrtf;
    }
    _ => {}
}

// There are certain architecture-specific implementations that are needed for correctness
// even with `arch` disabled. These are configured here.
cfg_select_nofmt! {
    x86_no_sse2 => {
        mod i586;
        pub use i586::{
            ceil,
            floor,
            rint,
            x87_exp,
            x87_exp10,
            x87_exp10f,
            x87_exp2,
            x87_exp2f,
            x87_expf,
        };
    }
    _ => {}
}
