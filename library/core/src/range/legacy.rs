//! # Legacy range types
//!
//! The types within this module will be replaced by the types
//! [`Range`], [`RangeInclusive`], and [`RangeFrom`] in the parent
//! module, [`core::range`].
//!
//! The types here are equivalent to those in [`core::ops`].

#[doc(inline)]
pub use crate::ops::{Range, RangeFrom, RangeInclusive};
