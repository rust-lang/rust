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
mod mips;
#[macro_use]
mod mips64;
#[macro_use]
mod loongarch;
#[macro_use]
mod s390x;

cfg_select! {
    any(target_arch = "x86", target_arch = "x86_64") => {
        pub use x86::__is_feature_detected;
        pub(crate) use x86::Feature;
    }
    target_arch = "arm" => {
        pub use arm::__is_feature_detected;
        pub(crate) use arm::Feature;
    }
    any(target_arch = "aarch64", target_arch = "arm64ec") => {
        pub use aarch64::__is_feature_detected;
        pub(crate) use aarch64::Feature;
    }
    any(target_arch = "riscv32", target_arch = "riscv64") => {
        pub use riscv::__is_feature_detected;
        pub(crate) use riscv::Feature;
    }
    any(target_arch = "powerpc", target_arch = "powerpc64") => {
        pub use powerpc::__is_feature_detected;
        pub(crate) use powerpc::Feature;
    }
    target_arch = "mips" => {
        pub use mips::__is_feature_detected;
        pub(crate) use mips::Feature;
    }
    target_arch = "mips64" => {
        pub use mips64::__is_feature_detected;
        pub(crate) use mips64::Feature;
    }
    any(target_arch = "loongarch32", target_arch = "loongarch64") => {
        pub use loongarch::__is_feature_detected;
        pub(crate) use loongarch::Feature;
    }
    target_arch = "s390x" => {
        pub use s390x::__is_feature_detected;
        pub(crate) use s390x::Feature;
    }
    _ => {
        // Unimplemented architecture:
        #[doc(hidden)]
        pub(crate) enum Feature {
            Null,
        }
        #[doc(hidden)]
        pub mod __is_feature_detected {}

        impl Feature {
            #[doc(hidden)]
            pub(crate) const ALL: &[Self] = &[];
            #[doc(hidden)]
            pub(crate) fn from_str(_s: &str) -> Result<Feature, ()> {
                Err(())
            }
            #[doc(hidden)]
            pub(crate) fn to_str(self) -> &'static str {
                ""
            }
        }
    }
}
