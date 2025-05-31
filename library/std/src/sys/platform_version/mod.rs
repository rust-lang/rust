//! Runtime lookup of operating system / platform version.
//!
//! Related to [RFC 3750](https://github.com/rust-lang/rfcs/pull/3750), which
//! does version detection at compile-time.
//!
//! See also the `os_info` crate.

#[cfg(target_vendor = "apple")]
mod darwin;

// FIXME(madsmtm): Use `RtlGetVersion` for version lookup on Windows.
// FIXME(madsmtm): Use `__system_property_get` for version lookup on Android.
