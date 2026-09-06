//! # Legacy range types
//!
//! The types within this module will be replaced by the types
//! [`Range`], [`RangeInclusive`], [`RangeToInclusive`], and [`RangeFrom`] in the parent
//! module, [`core::range`].
//!
//! The types here are equivalent to those in [`core::ops`].

#[doc(inline)]
#[stable(feature = "new_range_api_legacy", since = "1.98.0")]
#[allow(clippy::useless_attribute)]
#[allow(incompatible_reexport_stability)] // Path stability is tracked separately.
pub use crate::ops::{Range, RangeFrom, RangeInclusive, RangeToInclusive};
