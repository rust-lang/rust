//! Field Reflection

/// Type representing a field of a `struct`.
///
/// # Safety
///
/// Given a valid value of type `Self::Base`, there exists a valid value of type `Self::Type` at
/// byte offset `OFFSET`.
#[lang = "UnalignedField"]
#[unstable(feature = "field_projections", issue = "145383")]
pub unsafe trait UnalignedField: Sized {
    /// The type of the base where this field exists in.
    #[lang = "UnalignedFieldBase"]
    type Base: ?Sized;

    /// The type of the field.
    #[lang = "UnalignedFieldType"]
    type Type: ?Sized;

    /// The offset of the field in bytes.
    #[lang = "UnalignedFieldOFFSET"]
    const OFFSET: usize = crate::intrinsics::unaligned_field_offset::<Self>();
}

/// Type representing an aligned field of a `struct`.
///
/// # Safety
///
/// Given a well-aligned value of type `Self::Base`, the field at `Self::OFFSET` of type
/// `Self::Type` is well-aligned.
#[lang = "Field"]
#[unstable(feature = "field_projections", issue = "145383")]
pub unsafe trait Field: UnalignedField {}

/// Expands to the field representing type of the given field.
///
/// The container type may be a `struct` or a tuple.
///
/// The field may be a nested field (`field1.field2`), but not an array index.
/// The field must be visible to the call site.
#[unstable(feature = "field_projections", issue = "145383")]
#[allow_internal_unstable(builtin_syntax)]
pub macro field_of($Container:ty, $($fields:expr)+ $(,)?) {
    builtin # field_of($Container, $($fields)+)
}
