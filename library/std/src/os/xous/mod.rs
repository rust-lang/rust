#![stable(feature = "rust1", since = "1.0.0")]
#![doc(cfg(target_os = "xous"))]
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod ffi;

#[stable(feature = "rust1", since = "1.0.0")]
pub mod services;

/// A prelude for conveniently writing platform-specific code.
///
/// Includes all extension traits, and some important type definitions.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod prelude {
    #[doc(no_inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::ffi::{OsStrExt, OsStringExt};
}
