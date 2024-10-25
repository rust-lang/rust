cfg_if::cfg_if! {
    // Tier 1
    if #[cfg(any(target_os = "linux", target_os = "android"))] {
        mod linux;
        pub use linux::{fill_bytes, hashmap_random_keys};
    } else if #[cfg(target_os = "windows")] {
        mod windows;
        pub use windows::fill_bytes;
    } else if #[cfg(target_vendor = "apple")] {
        mod apple;
        pub use apple::fill_bytes;
    // Others, in alphabetical ordering.
    } else if #[cfg(any(
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "haiku",
        target_os = "illumos",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "rtems",
        target_os = "solaris",
        target_os = "vita",
    ))] {
        mod arc4random;
        pub use arc4random::fill_bytes;
    } else if #[cfg(target_os = "emscripten")] {
        mod getentropy;
        pub use getentropy::fill_bytes;
    } else if #[cfg(target_os = "espidf")] {
        mod espidf;
        pub use espidf::fill_bytes;
    } else if #[cfg(target_os = "fuchsia")] {
        mod fuchsia;
        pub use fuchsia::fill_bytes;
    } else if #[cfg(target_os = "hermit")] {
        mod hermit;
        pub use hermit::fill_bytes;
    } else if #[cfg(target_os = "horizon")] {
        // FIXME: add arc4random_buf to shim-3ds
        mod horizon;
        pub use horizon::fill_bytes;
    } else if #[cfg(any(
        target_os = "aix",
        target_os = "hurd",
        target_os = "l4re",
        target_os = "nto",
        target_os = "nuttx",
    ))] {
        mod unix_legacy;
        pub use unix_legacy::fill_bytes;
    } else if #[cfg(target_os = "redox")] {
        mod redox;
        pub use redox::fill_bytes;
    } else if #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))] {
        mod sgx;
        pub use sgx::fill_bytes;
    } else if #[cfg(target_os = "solid_asp3")] {
        mod solid;
        pub use solid::fill_bytes;
    } else if #[cfg(target_os = "teeos")] {
        mod teeos;
        pub use teeos::fill_bytes;
    } else if #[cfg(target_os = "uefi")] {
        mod uefi;
        pub use uefi::fill_bytes;
    } else if #[cfg(target_os = "vxworks")] {
        mod vxworks;
        pub use vxworks::fill_bytes;
    } else if #[cfg(target_os = "wasi")] {
        mod wasi;
        pub use wasi::fill_bytes;
    } else if #[cfg(target_os = "zkvm")] {
        mod zkvm;
        pub use zkvm::fill_bytes;
    } else if #[cfg(any(
        all(target_family = "wasm", target_os = "unknown"),
        target_os = "xous",
    ))] {
        // FIXME: finally remove std support for wasm32-unknown-unknown
        // FIXME: add random data generation to xous
        mod unsupported;
        pub use unsupported::{fill_bytes, hashmap_random_keys};
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    all(target_family = "wasm", target_os = "unknown"),
    target_os = "xous",
)))]
pub fn hashmap_random_keys() -> (u64, u64) {
    let mut buf = [0; 16];
    fill_bytes(&mut buf);
    let k1 = u64::from_ne_bytes(buf[..8].try_into().unwrap());
    let k2 = u64::from_ne_bytes(buf[8..].try_into().unwrap());
    (k1, k2)
}
