//! # Legacy range types
//!
//! The types within this module will be replaced by the types
//! [`Range`], [`RangeInclusive`], and [`RangeFrom`] in the parent
//! module, [`core::range`].
//!
//! The types here are equivalent to those in [`core::ops`].

#[doc(inline)]
#[unstable(feature = "new_range_api", issue = "125687")]
pub use crate::ops::Range;

#[doc(inline)]
#[unstable(feature = "new_range_api", issue = "125687")]
pub use crate::ops::RangeInclusive;

#[doc(inline)]
#[unstable(feature = "new_range_api", issue = "125687")]
pub use crate::ops::RangeFrom;
