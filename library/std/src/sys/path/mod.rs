cfg_if::cfg_if! {
    if #[cfg(target_os = "windows")] {
        mod windows;
        pub(crate) use windows::*;
    } else if #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))] {
        mod sgx;
        pub(crate) use sgx::*;
    } else if #[cfg(any(
        target_os = "uefi",
        target_os = "solid_asp3",
    ))] {
        mod unsupported_backslash;
        pub(crate) use unsupported_backslash::*;
    } else {
        mod unix;
        pub(crate) use unix::*;
    }
}
