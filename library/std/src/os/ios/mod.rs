//! iOS-specific definitions

#![stable(feature = "raw_ext", since = "1.1.0")]

#[stable(feature = "metadata_ext", since = "1.1.0")]
pub mod fs {
    #[stable(feature = "file_set_times", since = "1.75.0")]
    pub use crate::os::darwin::fs::FileTimesExt;
    #[stable(feature = "metadata_ext", since = "1.1.0")]
    pub use crate::os::darwin::fs::MetadataExt;
}

/// iOS-specific raw type definitions
#[stable(feature = "raw_ext", since = "1.1.0")]
#[deprecated(
    since = "1.8.0",
    note = "these type aliases are no longer supported by \
            the standard library, the `libc` crate on \
            crates.io should be used instead for the correct \
            definitions"
)]
#[allow(deprecated)]
pub mod raw {
    #[doc(inline)]
    #[stable(feature = "raw_ext", since = "1.1.0")]
    pub use crate::os::darwin::raw::*;
}
