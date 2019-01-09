//! HermitCore-specific raw type definitions

#![stable(feature = "raw_ext", since = "1.1.0")]
#![rustc_deprecated(since = "1.8.0",
                    reason = "these type aliases are no longer supported by \
                              the standard library, the `libc` crate on \
                              crates.io should be used instead for the correct \
                              definitions")]
#![allow(deprecated)]
#![allow(missing_debug_implementations)]

#[stable(feature = "pthread_t", since = "1.8.0")]
pub use libc::pthread_t;

#[doc(inline)]
#[stable(feature = "raw_ext", since = "1.1.0")]
pub use libc::{dev_t, mode_t, off_t, ino_t, nlink_t, blksize_t, blkcnt_t, stat, time_t};
