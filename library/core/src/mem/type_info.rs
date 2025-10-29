//! MVP for exposing compile-time information about types in a
//! runtime or const-eval processable way.

use crate::any::TypeId;
use crate::intrinsics::type_of;

/// Compile-time type information.
#[derive(Debug)]
#[non_exhaustive]
#[lang = "type_info"]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Type {
    /// Per-type information
    pub kind: TypeKind,
}

impl TypeId {
    /// Compute the type information of a concrete type.
    /// It can only be called at compile time.
    #[unstable(feature = "type_info", issue = "146922")]
    #[rustc_const_unstable(feature = "type_info", issue = "146922")]
    pub const fn info(self) -> Type {
        type_of(self)
    }
}

impl Type {
    /// Returns the type information of the generic type parameter.
    #[unstable(feature = "type_info", issue = "146922")]
    #[rustc_const_unstable(feature = "type_info", issue = "146922")]
    pub const fn of<T: 'static>() -> Self {
        const { TypeId::of::<T>().info() }
    }
}

/// Compile-time type information.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub enum TypeKind {
    /// Tuples.
    Tuple(Tuple),
    /// Primitives
    /// FIXME(#146922): disambiguate further
    Leaf,
    /// FIXME(#146922): add all the common types
    Other,
}

/// Compile-time type information about tuples.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Tuple {
    /// All fields of a tuple.
    pub fields: &'static [Field],
}

/// Compile-time type information about fields of tuples, structs and enum variants.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Field {
    /// The field's type.
    pub ty: TypeId,
    /// Offset in bytes from the parent type
    pub offset: usize,
}
