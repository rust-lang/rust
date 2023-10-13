//! Traits, helpers, and type definitions for core I/O functionality.

#![unstable(feature = "core_io", issue = "none")]

#[unstable(feature = "raw_os_error_ty", issue = "107792")]
pub use self::error::RawOsError;
#[unstable(feature = "core_io_error", issue = "none")]
#[cfg(target_has_atomic_load_store = "ptr")]
pub use self::error::{Error, ErrorKind, Result};

#[cfg(target_has_atomic_load_store = "ptr")]
#[unstable(feature = "core_io_error_internals", issue = "none")]
pub use error::const_io_error;

#[cfg(target_has_atomic_load_store = "ptr")]
#[unstable(feature = "core_io_error_internals", issue = "none")]
pub mod error_internals {
    //! implementation detail of [`Error`]

    pub use super::error::*;
}

#[cfg(target_has_atomic_load_store = "ptr")]
mod error;
