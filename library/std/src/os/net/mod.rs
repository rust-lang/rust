//! OS-specific networking functionality.

#[cfg(any(target_os = "linux", target_os = "android", doc))]
pub(super) mod linux_ext;
