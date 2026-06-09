//! Field Reflection

use crate::fmt;
use crate::hash::{Hash, Hasher};
use crate::marker::PhantomData;

/// Field Representing Type
#[unstable(feature = "field_representing_type_raw", issue = "none")]
#[lang = "field_representing_type"]
#[fundamental]
pub struct FieldRepresentingType<T: ?Sized, const VARIANT: u32, const FIELD: u32> {
    // We want this type to be invariant over `T`, because otherwise `field_of!(Struct<'short>,
    // field)` is a subtype of `field_of!(Struct<'long>, field)`. This subtype relationship does not
    // have an immediately obvious meaning and we want to prevent people from relying on it.
    _phantom: PhantomData<fn(T) -> T>,
}

impl<T: ?Sized, const VARIANT: u32, const FIELD: u32> fmt::Debug
    for FieldRepresentingType<T, VARIANT, FIELD>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        enum Member {
            Name(&'static str),
            Index(u32),
        }
        impl fmt::Display for Member {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    Self::Name(name) => fmt::Display::fmt(name, f),
                    Self::Index(idx) => fmt::Display::fmt(idx, f),
                }
            }
        }
        let (variant, field) = const {
            use crate::mem::type_info::{Type, TypeKind};
            match Type::of::<T>().kind {
                TypeKind::Struct(struct_) => {
                    (None, Member::Name(struct_.fields[FIELD as usize].name))
                }
                TypeKind::Tuple(_) => (None, Member::Index(FIELD)),
                TypeKind::Enum(enum_) => {
                    let variant = &enum_.variants[VARIANT as usize];
                    (Some(variant.name), Member::Name(variant.fields[FIELD as usize].name))
                }
                TypeKind::Union(union) => (None, Member::Name(union.fields[FIELD as usize].name)),
                _ => unreachable!(),
            }
        };
        let type_name = const { crate::any::type_name::<T>() };
        match variant {
            Some(variant) => write!(f, "field_of!({type_name}, {variant}.{field})"),
            None => write!(f, "field_of!({type_name}, {field})"),
        }
        // NOTE: if there are changes in the reflection work and the above no
        // longer compiles, then the following debug impl could also work in
        // the meantime:
        // ```rust
        // let type_name = const { type_name::<T>() };
        // write!(f, "field_of!({type_name}, {VARIANT}.{FIELD})")
        // ```
    }
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

impl<T: ?Sized, const VARIANT: u32, const FIELD: u32> Default
    for FieldRepresentingType<T, VARIANT, FIELD>
{
    fn default() -> Self {
        Self { _phantom: PhantomData::default() }
    }
}

impl<T: ?Sized, const VARIANT: u32, const FIELD: u32> Hash
    for FieldRepresentingType<T, VARIANT, FIELD>
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self._phantom.hash(state);
    }
}

impl<T: ?Sized, const VARIANT: u32, const FIELD: u32> PartialEq
    for FieldRepresentingType<T, VARIANT, FIELD>
{
    fn eq(&self, other: &Self) -> bool {
        self._phantom == other._phantom
    }
}

impl<T: ?Sized, const VARIANT: u32, const FIELD: u32> Eq
    for FieldRepresentingType<T, VARIANT, FIELD>
{
}

impl<T: ?Sized, const VARIANT: u32, const FIELD: u32> PartialOrd
    for FieldRepresentingType<T, VARIANT, FIELD>
{
    fn partial_cmp(&self, other: &Self) -> Option<crate::cmp::Ordering> {
        self._phantom.partial_cmp(&other._phantom)
    }
}

impl<T: ?Sized, const VARIANT: u32, const FIELD: u32> Ord
    for FieldRepresentingType<T, VARIANT, FIELD>
{
    fn cmp(&self, other: &Self) -> crate::cmp::Ordering {
        self._phantom.cmp(&other._phantom)
    }
}

/// Expands to the field representing type of the given field.
///
/// The container type may be a tuple, `struct`, `union` or `enum`. In the case of an enum, the
/// variant must also be specified. Only a single field is supported.
#[unstable(feature = "field_projections", issue = "145383")]
#[allow_internal_unstable(field_representing_type_raw, builtin_syntax)]
#[diagnostic::on_unmatch_args(
    note = "this macro expects a container type and a field path, like `field_of!(Type, field)` or `field_of!(Enum, Variant.field)`"
)]
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
