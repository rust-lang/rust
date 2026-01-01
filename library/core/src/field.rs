//! Field Reflection

/// Expands to the field representing type of the given field.
///
/// The container type may be a tuple, `struct`, `union` or `enum`. In the case of an enum, the
/// variant must also be specified. Only a single field is supported.
#[unstable(feature = "field_projections", issue = "145383")]
#[allow_internal_unstable(builtin_syntax)]
pub macro field_of($Container:ty, $($fields:expr)+ $(,)?) {
    builtin # field_of($Container, $($fields)+)
}
