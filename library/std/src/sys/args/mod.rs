//! Platform-dependent command line arguments abstraction.

#![forbid(unsafe_op_in_unsafe_fn)]

cfg_if::cfg_if! {
    if #[cfg(any(
        all(target_family = "unix", not(any(target_os = "espidf", target_os = "vita"))),
        target_os = "hermit",
    ))] {
        mod unix;
        pub use unix::*;
    } else if #[cfg(target_family = "windows")] {
        mod windows;
        pub use windows::*;
    } else if #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))] {
        mod sgx;
        pub use sgx::*;
    } else if #[cfg(target_os = "uefi")] {
        mod uefi;
        pub use uefi::*;
    } else if #[cfg(target_os = "wasi")] {
        mod wasi;
        pub use wasi::*;
    } else if #[cfg(target_os = "xous")] {
        mod xous;
        pub use xous::*;
    } else if #[cfg(target_os = "zkvm")] {
        mod zkvm;
        pub use zkvm::*;
    } else {
        mod unsupported;
        pub use unsupported::*;
    }
}
