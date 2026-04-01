//! MVP for exposing compile-time information about types in a
//! runtime or const-eval processable way.

use crate::any::TypeId;
use crate::intrinsics::{type_id, type_of};
use crate::marker::PointeeSized;
use crate::ptr::DynMetadata;

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

/// Info of a trait implementation, you can retrieve the vtable with [Self::get_vtable]
#[derive(Debug, PartialEq, Eq)]
#[unstable(feature = "type_info", issue = "146922")]
#[non_exhaustive]
pub struct TraitImpl<T: PointeeSized> {
    pub(crate) vtable: DynMetadata<T>,
}

impl<T: PointeeSized> TraitImpl<T> {
    /// Gets the raw vtable for type reflection mapping
    pub const fn get_vtable(&self) -> DynMetadata<T> {
        self.vtable
    }
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
    ///
    /// Note: Unlike `TypeId`s obtained via `TypeId::of`, the `Type`
    /// struct and its fields contain `TypeId`s that are not necessarily
    /// derived from types that outlive `'static`. This means that using
    /// the `TypeId`s (transitively) obtained from this function will
    /// be able to break invariants that other `TypeId` consuming crates
    /// may have assumed to hold.
    #[unstable(feature = "type_info", issue = "146922")]
    #[rustc_const_unstable(feature = "type_info", issue = "146922")]
    pub const fn of<T: ?Sized>() -> Self {
        const { type_id::<T>().info() }
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
    /// Structs.
    Struct(Struct),
    /// Enums.
    Enum(Enum),
    /// Unions.
    Union(Union),
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
    /// Function pointers.
    FnPtr(FnPtr),
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
    /// The name of the field.
    pub name: &'static str,
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

/// Compile-time type information about structs.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Struct {
    /// Instantiated generics of the struct.
    pub generics: &'static [Generic],
    /// All fields of the struct.
    pub fields: &'static [Field],
    /// Whether the struct field list is non-exhaustive.
    pub non_exhaustive: bool,
}

/// Compile-time type information about unions.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Union {
    /// Instantiated generics of the union.
    pub generics: &'static [Generic],
    /// All fields of the union.
    pub fields: &'static [Field],
}

/// Compile-time type information about enums.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Enum {
    /// Instantiated generics of the enum.
    pub generics: &'static [Generic],
    /// All variants of the enum.
    pub variants: &'static [Variant],
    /// Whether the enum variant list is non-exhaustive.
    pub non_exhaustive: bool,
}

/// Compile-time type information about variants of enums.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Variant {
    /// The name of the variant.
    pub name: &'static str,
    /// All fields of the variant.
    pub fields: &'static [Field],
    /// Whether the enum variant fields is non-exhaustive.
    pub non_exhaustive: bool,
}

/// Compile-time type information about instantiated generics of structs, enum and union variants.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub enum Generic {
    /// Lifetimes.
    Lifetime(Lifetime),
    /// Types.
    Type(GenericType),
    /// Const parameters.
    Const(Const),
}

/// Compile-time type information about generic lifetimes.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Lifetime {
    // No additional information to provide for now.
}

/// Compile-time type information about instantiated generic types.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct GenericType {
    /// The type itself.
    pub ty: TypeId,
}

/// Compile-time type information about generic const parameters.
#[derive(Debug)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
pub struct Const {
    /// The const's type.
    pub ty: TypeId,
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

#[derive(Debug)]
#[unstable(feature = "type_info", issue = "146922")]
/// Function pointer, e.g. fn(u8),
pub struct FnPtr {
    /// Unsafety, true is unsafe
    pub unsafety: bool,

    /// Abi, e.g. extern "C"
    pub abi: Abi,

    /// Function inputs
    pub inputs: &'static [TypeId],

    /// Function return type, default is TypeId::of::<()>
    pub output: TypeId,

    /// Vardiadic function, e.g. extern "C" fn add(n: usize, mut args: ...);
    pub variadic: bool,
}

#[derive(Debug, Default)]
#[non_exhaustive]
#[unstable(feature = "type_info", issue = "146922")]
/// Abi of [FnPtr]
pub enum Abi {
    /// Named abi, e.g. extern "custom", "stdcall" etc.
    Named(&'static str),

    /// Default
    #[default]
    ExternRust,

    /// C-calling convention
    ExternC,
}
