//! MVP for exposing compile-time information about types in a
//! runtime or const-eval processable way.

use crate::any::TypeId;
use crate::intrinsics::{type_id_implements_trait, type_id_is_trait, type_of};
use crate::ptr;

/// Compile-time type information.
#[derive(Debug)]
#[non_exhaustive]
#[lang = "type_info"]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Type {
    /// Per-type information
    pub kind: TypeKind,
    /// Size of the type. `None` if it is unsized
    pub size: Option<usize>,
}

impl TypeId {
    /// Compute the type information of a concrete type.
    /// It can only be called at compile time.
    #[unstable(feature = "type_info", issue = "146922")]
    #[rustc_const_unstable(feature = "type_info", issue = "146922")]
    pub const fn info(self) -> Type {
        type_of(self)
    }

    /// Checks if the type represented by the `TypeId` implements the trait.
    /// It can only be called at compile time.
    pub const fn has_trait<
        T: ptr::Pointee<Metadata = ptr::DynMetadata<T>> + ?Sized + 'static,
    >(
        self,
    ) -> bool {
        type_id_implements_trait(self, TypeId::of::<T>())
    }

    /// Checks if the type represented by the `TypeId` implements the trait represented by the secondary `TypeId`.
    /// Returns `None` if the `trait_represented_by_type_id` is not a trait represented by type id.
    /// It can only be called at compile time.
    pub const fn has_trait_represented_by_type_id(
        self,
        trait_represented_by_type_id: Self,
    ) -> Option<bool> {
        if type_id_is_trait(trait_represented_by_type_id) {
            Some(type_id_implements_trait(self, trait_represented_by_type_id))
        } else {
            None
        }
    }
}

impl Type {
    /// Returns the type information of the generic type parameter.
    #[unstable(feature = "type_info", issue = "146922")]
    #[rustc_const_unstable(feature = "type_info", issue = "146922")]
    // FIXME(reflection): don't require the 'static bound
    pub const fn of<T: ?Sized + 'static>() -> Self {
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
    /// Arrays.
    Array(Array),
    /// Slices.
    Slice(Slice),
    /// Dynamic Traits.
    DynTrait(DynTrait),
    /// Primitive boolean type.
    Bool(Bool),
    /// Primitive character type.
    Char(Char),
    /// Primitive signed and unsigned integer type.
    Int(Int),
    /// Primitive floating-point type.
    Float(Float),
    /// String slice type.
    Str(Str),
    /// References.
    Reference(Reference),
    /// Pointers.
    Pointer(Pointer),
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

/// Compile-time type information about arrays.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Array {
    /// The type of each element in the array.
    pub element_ty: TypeId,
    /// The length of the array.
    pub len: usize,
}

/// Compile-time type information about slices.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Slice {
    /// The type of each element in the slice.
    pub element_ty: TypeId,
}

/// Compile-time type information about dynamic traits.
/// FIXME(#146922): Add super traits and generics
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct DynTrait {
    /// The predicates of  a dynamic trait.
    pub predicates: &'static [DynTraitPredicate],
}

/// Compile-time type information about a dynamic trait predicate.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct DynTraitPredicate {
    /// The type of the trait as a dynamic trait type.
    pub trait_ty: Trait,
}

/// Compile-time type information about a trait.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Trait {
    /// The TypeId of the trait as a dynamic type
    pub ty: TypeId,
    /// Whether the trait is an auto trait
    pub is_auto: bool,
}

/// Compile-time type information about `bool`.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Bool {
    // No additional information to provide for now.
}

/// Compile-time type information about `char`.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Char {
    // No additional information to provide for now.
}

/// Compile-time type information about signed and unsigned integer types.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Int {
    /// The bit width of the signed integer type.
    pub bits: u32,
    /// Whether the integer type is signed.
    pub signed: bool,
}

/// Compile-time type information about floating-point types.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Float {
    /// The bit width of the floating-point type.
    pub bits: u32,
}

/// Compile-time type information about string slice types.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Str {
    // No additional information to provide for now.
}

/// Compile-time type information about references.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Reference {
    /// The type of the value being referred to.
    pub pointee: TypeId,
    /// Whether this reference is mutable or not.
    pub mutable: bool,
}

/// Compile-time type information about pointers.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Pointer {
    /// The type of the value being pointed to.
    pub pointee: TypeId,
    /// Whether this pointer is mutable or not.
    pub mutable: bool,
}
