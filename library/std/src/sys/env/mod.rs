//! Platform-dependent environment variables abstraction.

#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(any(
    target_family = "unix",
    target_os = "hermit",
    all(target_vendor = "fortanix", target_env = "sgx"),
    target_os = "solid_asp3",
    target_os = "uefi",
    target_os = "wasi",
    target_os = "xous",
))]
mod common;

cfg_select! {
    target_family = "unix" => {
        mod unix;
        pub use unix::*;
    }
    target_family = "windows" => {
        mod windows;
        pub use windows::*;
    }
    target_os = "hermit" => {
        mod hermit;
        pub use hermit::*;
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
        pub use sgx::*;
    }
    target_os = "solid_asp3" => {
        mod solid;
        pub use solid::*;
    }
    target_os = "uefi" => {
        mod uefi;
        pub use uefi::*;
    }
    target_os = "wasi" => {
        mod wasi;
        pub use wasi::*;
    }
    target_os = "xous" => {
        mod xous;
        pub use xous::*;
    }
    target_os = "zkvm" => {
        mod zkvm;
        pub use zkvm::*;
    }
    _ => {
        mod unsupported;
        pub use unsupported::*;
    }
}
