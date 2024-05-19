//! OS-specific networking functionality.

// See cfg macros in `library/std/src/os/mod.rs` for why these platforms must
// be special-cased during rustdoc generation.
#[cfg(not(all(
    doc,
    any(
        all(target_arch = "wasm32", not(target_os = "wasi")),
        all(target_vendor = "fortanix", target_env = "sgx")
    )
)))]
#[cfg(any(target_os = "linux", target_os = "android", doc))]
pub(super) mod linux_ext;
