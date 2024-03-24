//! ASN.1 `CHOICE` support.

use crate::{Decode, FixedTag, Tag, Tagged};

/// ASN.1 `CHOICE` denotes a union of one or more possible alternatives.
///
/// The types MUST have distinct tags.
///
/// This crate models choice as a trait, with a blanket impl for all types
/// which impl `Decode + FixedTag` (i.e. they are modeled as a `CHOICE`
/// with only one possible variant)
pub trait Choice<'a>: Decode<'a> + Tagged {
    /// Is the provided [`Tag`] decodable as a variant of this `CHOICE`?
    fn can_decode(tag: Tag) -> bool;
}

/// This blanket impl allows any [`Tagged`] type to function as a [`Choice`]
/// with a single alternative.
impl<'a, T> Choice<'a> for T
where
    T: Decode<'a> + FixedTag,
{
    fn can_decode(tag: Tag) -> bool {
        T::TAG == tag
    }
}
