//! Platform-dependent command line arguments abstraction.

#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(any(
    all(target_family = "unix", not(any(target_os = "espidf", target_os = "vita"))),
    target_family = "windows",
    target_os = "hermit",
    target_os = "uefi",
    target_os = "wasi",
    target_os = "xous",
))]
mod common;

cfg_select! {
    any(
        all(target_family = "unix", not(any(target_os = "espidf", target_os = "vita"))),
        target_os = "hermit",
    ) => {
        mod unix;
        pub use unix::*;
    }
    target_family = "windows" => {
        mod windows;
        pub use windows::*;
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
        pub use sgx::*;
    }
    target_os = "uefi" => {
        mod uefi;
        pub use uefi::*;
    }
    all(target_os = "wasi", target_env = "p1") => {
        mod wasip1;
        pub use wasip1::*;
    }
    all(target_os = "wasi", target_env = "p2") => {
        mod wasip2;
        pub use wasip2::*;
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
