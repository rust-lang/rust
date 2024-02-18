cfg_if::cfg_if! {
    if #[cfg(target_os = "windows")] {
        mod windows;
        pub use windows::*;
    } else if #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))] {
        mod sgx;
        pub use sgx::*;
    } else if #[cfg(any(
        target_os = "uefi",
        target_os = "solid_asp3",
    ))] {
        mod unsupported_backslash;
        pub use unsupported_backslash::*;
    } else {
        mod unix;
        pub use unix::*;
    }
}
