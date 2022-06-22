cfg_if::cfg_if! {
    if #[cfg(any(
        target_os = "linux",
        target_os = "android",
        all(target_arch = "wasm32", target_feature = "atomics"),
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "dragonfly",
    ))] {
        mod futex;
        pub use futex::Parker;
    } else if #[cfg(windows)] {
        pub use crate::sys::thread_parker::Parker;
    } else if #[cfg(target_family = "unix")] {
        pub use crate::sys::thread_parker::Parker;
    } else if #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))] {
        pub use crate::sys::thread_parker::Parker;
    } else {
        mod generic;
        pub use generic::Parker;
    }
}
