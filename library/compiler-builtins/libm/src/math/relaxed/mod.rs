//! Relaxed precision implementations.
//!
//! These functions may be smaller or faster than those in the main `math` module, but will
//! not be as accurate.

mod cbrtf64;
mod hypotf64;

pub use cbrtf64::cbrtf64;
pub use hypotf64::hypotf64;
