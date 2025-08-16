#![allow(dead_code)]

// Export the macros for all supported architectures.
#[macro_use]
mod x86;
#[macro_use]
mod arm;
#[macro_use]
mod aarch64;
#[macro_use]
mod riscv;
#[macro_use]
mod powerpc;
#[macro_use]
mod powerpc64;
#[macro_use]
mod mips;
#[macro_use]
mod mips64;
#[macro_use]
mod loongarch;
#[macro_use]
mod s390x;

cfg_select! {
    any(target_arch = "x86", target_arch = "x86_64") => {
        #[stable(feature = "simd_x86", since = "1.27.0")]
        pub use x86::*;
    }
    target_arch = "arm" => {
        #[unstable(feature = "stdarch_arm_feature_detection", issue = "111190")]
        pub use arm::*;
    }
    any(target_arch = "aarch64", target_arch = "arm64ec") => {
        #[stable(feature = "simd_aarch64", since = "1.60.0")]
        pub use aarch64::*;
    }
    any(target_arch = "riscv32", target_arch = "riscv64") => {
        #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")]
        pub use riscv::*;
    }
    target_arch = "powerpc" => {
        #[unstable(feature = "stdarch_powerpc_feature_detection", issue = "111191")]
        pub use powerpc::*;
    }
    target_arch = "powerpc64" => {
        #[unstable(feature = "stdarch_powerpc_feature_detection", issue = "111191")]
        pub use powerpc64::*;
    }
    target_arch = "mips" => {
        #[unstable(feature = "stdarch_mips_feature_detection", issue = "111188")]
        pub use mips::*;
    }
    target_arch = "mips64" => {
        #[unstable(feature = "stdarch_mips_feature_detection", issue = "111188")]
        pub use mips64::*;
    }
    any(target_arch = "loongarch32", target_arch = "loongarch64") => {
        #[stable(feature = "stdarch_loongarch_feature", since = "1.89.0")]
        pub use loongarch::*;
    }
    target_arch = "s390x" => {
        #[unstable(feature = "stdarch_s390x_feature_detection", issue = "135413")]
        pub use s390x::*;
    }
    _ => {
        // Unimplemented architecture:
        #[doc(hidden)]
        pub(crate) enum Feature {
            Null
        }
        #[doc(hidden)]
        #[unstable(feature = "stdarch_internal", issue = "none")]
        pub mod __is_feature_detected {}

        impl Feature {
            #[doc(hidden)]
            pub(crate) fn from_str(_s: &str) -> Result<Feature, ()> { Err(()) }
            #[doc(hidden)]
            pub(crate) fn to_str(self) -> &'static str { "" }
        }
    }
}
