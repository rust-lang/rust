//! Runtime lookup of operating system / platform version.
//!
//! Related to [RFC 3750](https://github.com/rust-lang/rfcs/pull/3750), which
//! does version detection at compile-time.
//!
//! See also the `os_info` crate.

#[cfg(target_vendor = "apple")]
mod darwin;

// In the future, we could expand this module with:
// - `RtlGetVersion` on Windows.
// - `__system_property_get` on Android.
