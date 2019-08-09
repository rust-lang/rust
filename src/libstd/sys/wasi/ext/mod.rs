pub mod ffi;
pub mod fs;
pub mod io;

/// A prelude for conveniently writing platform-specific code.
///
/// Includes all extension traits, and some important type definitions.
#[stable(feature = "rust1", since = "1.0.0")]
pub mod prelude {
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use crate::sys::ext::ffi::{OsStringExt, OsStrExt};
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use crate::sys::ext::fs::{FileExt, DirEntryExt, MetadataExt, OpenOptionsExt};
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use crate::sys::ext::fs::FileTypeExt;
    #[doc(no_inline)] #[stable(feature = "rust1", since = "1.0.0")]
    pub use crate::sys::ext::io::{AsRawFd, IntoRawFd, FromRawFd};
}
