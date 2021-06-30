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

/// Converts an enum from its underlying repr.
///
/// # Safety
///
/// This trait must only be implemented for types which are transmutable from their repr for
/// inhabited repr values. Callers may assume that it is safe to transmute an instance of `Repr`
/// into its associated `Repr` type if that value is inhabited for this type.
#[unstable(feature = "enum_as_repr", issue = "86772")]
pub unsafe trait FromRepr: AsRepr {
    /// Tries to convert an enum from its underlying repr type.
    fn try_from_repr(from: Self::Repr) -> Result<Self, TryFromReprError<Self::Repr>>;

    /// Converts from the enum's underlying repr type to this enum.
    ///
    /// # Safety
    ///
    /// This is only safe to call if it is known that the value being converted has a matching
    /// variant of this enum. Attempting to convert a value which doesn't correspond to an enum
    /// variant causes undefined behavior.
    unsafe fn from_repr(from: Self::Repr) -> Self {
        // SAFETY: Guaranteed to be safe from the safety constraints of the unsafe trait itself.
        let value = unsafe { crate::mem::transmute_copy(&from) };
        drop(from);
        value
    }
}

/// Derive macro generating an impl of the trait `FromRepr` for enums.
#[cfg(not(bootstrap))]
#[rustc_builtin_macro]
#[unstable(feature = "enum_as_repr", issue = "86772")]
pub macro FromRepr($item:item) {
    /* compiler built-in */
}

/// The error type returned when a checked integral type conversion fails.
/// Ideally this would be the same as `core::num::TryFromIntError` but it's not publicly
/// constructable.
#[unstable(feature = "enum_as_repr", issue = "86772")]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TryFromReprError<T: Sized>(pub T);
