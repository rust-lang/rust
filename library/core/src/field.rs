//! Field Reflection

use crate::marker::PhantomData;

/// Field Representing Type
#[unstable(feature = "field_representing_type_raw", issue = "none")]
#[lang = "field_representing_type"]
#[expect(missing_debug_implementations)]
#[fundamental]
pub struct FieldRepresentingType<T: ?Sized, const VARIANT: u32, const FIELD: u32> {
    _phantom: PhantomData<T>,
}

// SAFETY: `FieldRepresentingType` doesn't contain any `T`
unsafe impl<T: ?Sized, const VARIANT: u32, const FIELD: u32> Send
    for FieldRepresentingType<T, VARIANT, FIELD>
{
}

// SAFETY: `FieldRepresentingType` doesn't contain any `T`
unsafe impl<T: ?Sized, const VARIANT: u32, const FIELD: u32> Sync
    for FieldRepresentingType<T, VARIANT, FIELD>
{
}

impl<T: ?Sized, const VARIANT: u32, const FIELD: u32> Copy
    for FieldRepresentingType<T, VARIANT, FIELD>
{
}

impl<T: ?Sized, const VARIANT: u32, const FIELD: u32> Clone
    for FieldRepresentingType<T, VARIANT, FIELD>
{
    fn clone(&self) -> Self {
        *self
    }
}

/// Expands to the field representing type of the given field.
///
/// The container type may be a tuple, `struct`, `union` or `enum`. In the case of an enum, the
/// variant must also be specified. Only a single field is supported.
#[unstable(feature = "field_projections", issue = "145383")]
#[allow_internal_unstable(field_representing_type_raw, builtin_syntax)]
// NOTE: when stabilizing this macro, we can never add new trait impls for `FieldRepresentingType`,
// since it is `#[fundamental]` and thus could break users of this macro, since the compiler expands
// it to `FieldRepresentingType<...>`. Thus stabilizing this requires careful thought about the
// completeness of the trait impls for `FieldRepresentingType`.
pub macro field_of($Container:ty, $($fields:expr)+ $(,)?) {
    builtin # field_of($Container, $($fields)+)
}

/// Type representing a field of a `struct`, `union`, `enum` variant or tuple.
///
/// # Safety
///
/// Given a valid value of type `Self::Base`, there exists a valid value of type `Self::Type` at
/// byte offset `OFFSET`
#[lang = "field"]
#[unstable(feature = "field_projections", issue = "145383")]
#[rustc_deny_explicit_impl]
#[rustc_dyn_incompatible_trait]
// NOTE: the compiler provides the impl of `Field` for `FieldRepresentingType` when it can guarantee
// the safety requirements of this trait. It also has to manually add the correct trait bounds on
// associated types (and the `Self` type). Thus any changes to the bounds here must be reflected in
// the old and new trait solver:
// - `fn assemble_candidates_for_field_trait` in
//   `compiler/rustc_trait_selection/src/traits/select/candidate_assembly.rs`, and
// - `fn consider_builtin_field_candidate` in
//   `compiler/rustc_next_trait_solver/src/solve/trait_goals.rs`.
pub unsafe trait Field: Send + Sync + Copy {
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
