//! Field Reflection

/// Type representing a field of a `struct` or tuple.
///
/// # Safety
///
/// Given a valid value of type `Self::Base`, there exists a valid value of type `Self::Type` at
/// byte offset `OFFSET`.
#[lang = "Field"]
#[unstable(feature = "field_projections", issue = "145383")]
#[rustc_deny_explicit_impl]
#[rustc_do_not_implement_via_object]
pub unsafe trait Field: Sized {
    /// The type of the base where this field exists in.
    #[lang = "FieldBase"]
    type Base;

    /// The type of the field.
    #[lang = "FieldType"]
    type Type;

    /// The offset of the field in bytes.
    #[lang = "FieldOffset"]
    const OFFSET: usize = crate::intrinsics::field_offset::<Self>();
}

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
