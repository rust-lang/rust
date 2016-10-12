#![stable(feature = "raw_ext", since = "1.1.0")]
#![rustc_deprecated(since = "1.8.0",
                    reason = "these type aliases are no longer supported by \
                              the standard library, the `libc` crate on \
                              crates.io should be used instead for the correct \
                              definitions")]

#[stable(feature = "raw_ext", since = "1.1.0")] pub struct blkcnt_t;
#[stable(feature = "raw_ext", since = "1.1.0")] pub struct blksize_t;
#[stable(feature = "raw_ext", since = "1.1.0")] pub struct dev_t;
#[stable(feature = "raw_ext", since = "1.1.0")] pub struct ino_t;
#[stable(feature = "raw_ext", since = "1.1.0")] pub struct mode_t;
#[stable(feature = "raw_ext", since = "1.1.0")] pub struct nlink_t;
#[stable(feature = "raw_ext", since = "1.1.0")] pub struct off_t;
#[stable(feature = "pthread_t", since = "1.8.0")]pub use super::libc::pthread_t;
#[stable(feature = "raw_ext", since = "1.1.0")] pub struct time_t;