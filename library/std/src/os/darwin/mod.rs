//! Platform-specific extensions to `std` for Darwin / Apple platforms.
//!
//! This is available on the following operating systems:
//! - macOS
//! - iOS
//! - tvOS
//! - watchOS
//! - visionOS
//!
//! Note: This module is called "Darwin" as that's the name of the underlying
//! core OS of the above operating systems, but it should not be confused with
//! the `-darwin` suffix in the `x86_64-apple-darwin` and
//! `aarch64-apple-darwin` target names, which are mostly named that way for
//! legacy reasons.

#![stable(feature = "os_darwin", since = "CURRENT_RUSTC_VERSION")]
#![doc(cfg(target_vendor = "apple"))]

pub mod fs;
// deprecated, but used for public reexport under `std::os::unix::raw`, as
// well as `std::os::macos`/`std::os::ios`, because those modules precede the
// decision to remove these.
pub(super) mod raw;
