//! Field Reflection

/// Type representing a field of a `struct`, `union`, `enum` variant or tuple.
///
/// # Safety
///
/// Given a valid value of type `Self::Base`, there exists a valid value of type `Self::Type` at
/// byte offset `OFFSET`.
#[lang = "field"]
#[unstable(feature = "field_projections", issue = "145383")]
#[rustc_deny_explicit_impl]
#[rustc_do_not_implement_via_object]
pub unsafe trait Field: Sized {
    /// The type of the base where this field exists in.
    #[lang = "field_base"]
    type Base;

    /// The type of the field.
    #[lang = "field_type"]
    type Type;

    /// The offset of the field in bytes.
    #[lang = "field_offset"]
    const OFFSET: usize = crate::intrinsics::field_offset::<Self>();
}

/// Expands to the field representing type of the given field.
///
/// The container type may be a tuple, `struct`, `union` or `enum`. In the case of an enum, the
/// variant must also be specified. Only a single field is supported.
#[unstable(feature = "field_projections", issue = "145383")]
#[allow_internal_unstable(builtin_syntax)]
pub macro field_of($Container:ty, $($fields:expr)+ $(,)?) {
    builtin # field_of($Container, $($fields)+)
}
