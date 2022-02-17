#![allow(dead_code)]

use cfg_if::cfg_if;

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

cfg_if! {
    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        pub use x86::*;
    } else if #[cfg(target_arch = "arm")] {
        pub use arm::*;
    } else if #[cfg(target_arch = "aarch64")] {
        pub use aarch64::*;
    } else if #[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))] {
        pub use riscv::*;
    } else if #[cfg(target_arch = "powerpc")] {
        pub use powerpc::*;
    } else if #[cfg(target_arch = "powerpc64")] {
        pub use powerpc64::*;
    } else if #[cfg(target_arch = "mips")] {
        pub use mips::*;
    } else if #[cfg(target_arch = "mips64")] {
        pub use mips64::*;
    } else {
        // Unimplemented architecture:
        #[doc(hidden)]
        pub(crate) enum Feature {
            Null
        }
        #[doc(hidden)]
        pub mod __is_feature_detected {}

        impl Feature {
            #[doc(hidden)]
            pub(crate) fn from_str(_s: &str) -> Result<Feature, ()> { Err(()) }
            #[doc(hidden)]
            pub(crate) fn to_str(self) -> &'static str { "" }
        }
    }
}
