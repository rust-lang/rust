//! Fuchsia-specific definitions

#![stable(feature = "raw_ext", since = "1.1.0")]

pub mod fs;
#[unstable(feature = "fuchsia_exit_status", issue = "none")]
pub mod process;
pub mod raw;
