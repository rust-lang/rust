//! Newtypes for working with text sizes/ranges in a more type-safe manner.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations, missing_docs)]

mod range;
mod size;
mod traits;

#[cfg(feature = "serde")]
mod serde_impls;

pub use crate::{range::TextRange, size::TextSize, traits::TextSized};

#[cfg(feature = "deepsize")]
deepsize::known_deep_size!(0, TextSize, TextRange);
