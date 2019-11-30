#![stable(feature = "rust1", since = "1.0.0")]
#![allow(missing_docs)]

pub mod ffi;
pub mod fs;
pub mod io;
pub mod process;
pub mod raw;

#[stable(feature = "rust1", since = "1.0.0")]
pub mod prelude {
    #[doc(no_inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::ffi::{OsStrExt, OsStringExt};
    #[doc(no_inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt};
    #[doc(no_inline)]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::io::{AsRawFd, FromRawFd, IntoRawFd, RawFd};
}
