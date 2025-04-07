//! Platform-dependent file descriptor abstraction.

#![forbid(unsafe_op_in_unsafe_fn)]

cfg_if::cfg_if! {
    if #[cfg(target_family = "unix")] {
        mod unix;
        pub use unix::*;
    } else if #[cfg(target_os = "hermit")] {
        mod hermit;
        pub use hermit::*;
    } else if #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))] {
        mod sgx;
        pub use sgx::*;
    } else if #[cfg(target_os = "wasi")] {
        mod wasi;
        pub use wasi::*;
    }
}
