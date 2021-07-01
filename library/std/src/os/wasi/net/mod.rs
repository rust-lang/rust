//! WASI-specific networking functionality

#![unstable(feature = "wasi_ext", issue = "71213")]

mod raw_fd;

#[unstable(feature = "wasi_ext", issue = "71213")]
pub use self::raw_fd::*;
