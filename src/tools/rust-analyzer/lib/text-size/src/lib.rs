//! Newtypes for working with text sizes/ranges in a more type-safe manner.
//!
//! Minimal Supported Rust Version: latest stable.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations, missing_docs)]

mod range;
mod size;
mod traits;

#[cfg(feature = "serde")]
mod serde_impls;

pub use crate::{range::TextRange, size::TextSize, traits::LenTextSize};

#[cfg(target_pointer_width = "16")]
compile_error!("text-size assumes usize >= u32 and does not work on 16-bit targets");
