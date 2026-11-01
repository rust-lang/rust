//! Platform-specific extensions to `core` for Darwin / Apple platforms.
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

#![unstable(feature = "darwin_objc", issue = "145496")]
#![doc(cfg(target_vendor = "apple"))]

pub mod objc;
