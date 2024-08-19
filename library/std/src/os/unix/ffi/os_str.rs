// Note: this file is currently reused in other `std::os::{platform}::ffi` modules to reduce duplication.
// Keep this in mind when applying changes to this file that only apply to `unix`.

#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc::ffi::os_str::os_str_ext_unix::OsStringExt;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::ffi::os_str::os_str_ext_unix::OsStrExt;
