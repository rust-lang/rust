//! Field Reflection

/// Type representing a (possibly unaligned) field of a `struct` or tuple.
///
/// # Safety
///
/// Given a valid value of type `Self::Base`, there exists a valid value of type `Self::Type` at
/// byte offset `OFFSET`.
#[lang = "UnalignedField"]
#[unstable(feature = "field_projections", issue = "145383")]
#[rustc_deny_explicit_impl]
#[rustc_do_not_implement_via_object]
pub unsafe trait UnalignedField: Sized {
    /// The type of the base where this field exists in.
    #[lang = "UnalignedFieldBase"]
    type Base: ?Sized;

    /// The type of the field.
    #[lang = "UnalignedFieldType"]
    type Type: ?Sized;

    /// The offset of the field in bytes.
    #[lang = "UnalignedFieldOFFSET"]
    const OFFSET: usize;
}

/// Type representing an aligned field of a `struct` or tuple.
///
/// # Safety
///
/// Given a well-aligned value of type `Self::Base`, the field at `Self::OFFSET` of type
/// `Self::Type` is well-aligned.
#[lang = "Field"]
#[unstable(feature = "field_projections", issue = "145383")]
#[rustc_deny_explicit_impl]
#[rustc_do_not_implement_via_object]
pub unsafe trait Field: UnalignedField {}
