//! OS-specific functionality.
#![unstable(feature = "core_os", issue = "none")]
#![allow(missing_docs)]

#[cfg(all(
    doc,
    any(
        all(target_family = "wasm", not(target_os = "wasi")),
        all(target_vendor = "fortanix", target_env = "sgx")
    )
))]
#[unstable(issue = "none", feature = "std_internals")]
pub mod darwin {}

// darwin
#[cfg(not(all(
    doc,
    any(
        all(target_family = "wasm", not(target_os = "wasi")),
        all(target_vendor = "fortanix", target_env = "sgx")
    )
)))]
#[cfg(any(target_vendor = "apple", doc))]
pub mod darwin;

#[cfg(any(target_family = "windows", doc))]
pub mod windows;
