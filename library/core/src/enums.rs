//! For introspecting and converting between C-like enums and numbers.

use core::marker::DiscriminantKind;
use core::mem::Discriminant;

/// Converts an enum to its underlying discriminant.
///
/// This trait is automatically implemented for all C-like enum types which have an explicit
/// numeric repr. It may not be manually implemented.
#[unstable(feature = "enum_as_repr", issue = "86772")]
#[cfg_attr(not(bootstrap), lang = "AsRepr")]
pub trait AsRepr: Sized {
    /// The underlying repr type of the enum.
    type Repr;

    /// Convert the enum to its underlying discriminant.
    fn as_repr(&self) -> Self::Repr;
}

/// Marker trait used by the trait solver to enable implementations of AsRepr.
/// This trait cannot be manually implemented outside of `core`.
#[unstable(feature = "enum_as_repr", issue = "86772")]
#[cfg_attr(not(bootstrap), lang = "HasAsReprImpl")]
pub trait HasAsReprImpl {}

impl<T: HasAsReprImpl> AsRepr for T {
    type Repr = <T as DiscriminantKind>::Discriminant;

    fn as_repr(&self) -> Self::Repr {
        core::intrinsics::discriminant_value(self)
    }
}

impl<T: HasAsReprImpl> AsRepr for Discriminant<T> {
    type Repr = <T as DiscriminantKind>::Discriminant;

    fn as_repr(&self) -> Self::Repr {
        self.discriminant()
    }
}
