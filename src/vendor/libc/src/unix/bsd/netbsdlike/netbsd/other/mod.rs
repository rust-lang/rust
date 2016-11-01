cfg_if! {
    if #[cfg(target_arch = "x86_64")] {
        mod b64;
        pub use self::b64::*;
    } else if #[cfg(any(target_arch = "arm",
                        target_arch = "powerpc",
                        target_arch = "x86"))] {
        mod b32;
        pub use self::b32::*;
    } else {
        // Unknown target_arch
    }
}
