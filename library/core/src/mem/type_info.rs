//! MVP for exposing compile-time information about types in a
//! runtime or const-eval processable way.

use crate::any::TypeId;
use crate::fmt;
use crate::intrinsics::{self, type_id, type_of};
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
    /// Whether the enum variant fields are non-exhaustive.
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

    // FIXME(splat): should these fields be private, or merged into an Option<u16>?
    /// Is any function argument splatted?
    pub is_splatted: bool,

    /// The index of the splatted function argument in `inputs`, only valid if `is_splatted` is true.
    /// e.g. in `fn overload(a: u8, #[splat] b: (f32, usize))` the index is 1, and it can be called
    /// as `overload(a, 1.0, 2)`.
    pub splatted_index: u16,
}

impl FnPtr {
    /// Returns the splatted function argument index, or `None` if no argument is splatted.
    pub const fn splatted(&self) -> Option<u16> {
        if self.is_splatted { Some(self.splatted_index) } else { None }
    }
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

impl TypeId {
    /// Returns the size of the type represented by this `TypeId`. `None` if it is unsized.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(type_info)]
    /// use std::any::TypeId;
    ///
    /// assert_eq!(const { TypeId::of::<u32>().size() }, Some(4));
    /// assert_eq!(const { TypeId::of::<[u8; 16]>().size() }, Some(16));
    /// ```
    #[unstable(feature = "type_info", issue = "146922")]
    #[rustc_const_unstable(feature = "type_info", issue = "146922")]
    #[rustc_comptime]
    pub fn size(self) -> Option<usize> {
        intrinsics::size_of_type_id(self)
    }

    /// Returns the number of variants of the type represented by this `TypeId`.
    ///
    /// For enums, this is the number of variants. For structs and unions, this is always 1.
    ///
    /// ```
    /// #![feature(type_info)]
    /// use std::any::TypeId;
    ///
    /// assert_eq!(const { TypeId::of::<Option<()>>().variants() }, 2);
    ///
    /// struct Unit;
    /// struct Point {
    ///     x: u32,
    ///     y: u32,
    /// }
    /// assert_eq!(const { TypeId::of::<Unit>().variants() }, 1);
    /// assert_eq!(const { TypeId::of::<Point>().variants() }, 1);
    /// assert_eq!(const { TypeId::of::<(f32, f32)>().variants() }, 1);
    /// ```
    #[unstable(feature = "type_info", issue = "146922")]
    #[rustc_const_unstable(feature = "type_info", issue = "146922")]
    #[rustc_comptime]
    pub fn variants(self) -> usize {
        intrinsics::type_id_variants(self)
    }

    /// Returns the number of fields at the given `variant_index` of the type represented by this `TypeId`.
    ///
    /// ```
    /// #![feature(type_info)]
    /// use std::any::TypeId;
    ///
    /// assert_eq!(const { TypeId::of::<u32>().fields(0) }, 0);
    ///
    /// struct Point {
    ///     x: u32,
    ///     y: u32,
    /// }
    /// assert_eq!(const { TypeId::of::<Point>().fields(0) }, 2);
    ///
    /// enum Enum {
    ///     Unit,
    ///     Tuple(u32, u64),
    ///     Struct { x: u32, y: u32, z: String },
    /// }
    /// assert_eq!(const { TypeId::of::<Enum>().fields(0) }, 0);
    /// assert_eq!(const { TypeId::of::<Enum>().fields(1) }, 2);
    /// assert_eq!(const { TypeId::of::<Enum>().fields(2) }, 3);
    /// ```
    ///
    /// The variant index refers to the source order index of a variant in a type.
    ///
    /// For enums, these are always `0..variant_count`, regardless of any custom discriminants that may have been defined.
    /// `struct`s, `tuples`, and `unions`s are considered to have a single variant with variant index zero.
    ///
    /// ```
    /// enum Number {
    ///     Seven = 7, // variant index == 0
    ///     Six = 6,   // variant index == 1
    /// }
    /// ```
    ///
    /// Out-of-bounds indexing will be treated as a compile-time error.
    ///
    /// ```compile_fail,E0080
    /// # #![feature(type_info)]
    /// # use std::any::TypeId;
    /// #
    /// # struct Point {
    /// #     x: u32,
    /// #     y: u32,
    /// # }
    /// # enum Enum {
    /// #     Unit,
    /// #     Tuple(u32, u64),
    /// #     Struct { x: u32, y: u32, z: String },
    /// # }
    /// const {
    ///     _ = TypeId::of::<Point>().fields(10); // error: indexing out of bounds: the len is 2 but the index is 10
    ///     _ = TypeId::of::<Enum>().fields(10); // error: indexing out of bounds: the len is 3 but the index is 10
    /// }
    /// ```
    #[unstable(feature = "type_info", issue = "146922")]
    #[rustc_const_unstable(feature = "type_info", issue = "146922")]
    #[rustc_comptime]
    pub fn fields(self, variant_index: usize) -> usize {
        intrinsics::type_id_fields(self, variant_index)
    }

    /// Returns the field representing type at the given index of the type represented by this `TypeId`.
    ///
    /// ```
    /// #![feature(type_info)]
    /// use std::any::TypeId;
    ///
    /// struct Point {
    ///     x: u32,
    ///     y: u32,
    /// }
    /// assert_eq!(const { TypeId::of::<Point>().field(0, 0).type_id() }, TypeId::of::<u32>());
    /// assert_eq!(const { TypeId::of::<Point>().field(0, 1).type_id() }, TypeId::of::<u32>());
    ///
    /// enum Enum {
    ///     Unit,
    ///     Tuple(u32, u64),
    ///     Struct { x: u32, y: u32, z: String },
    /// }
    /// assert_eq!(const { TypeId::of::<Enum>().field(1, 0).type_id() }, TypeId::of::<u32>());
    /// assert_eq!(const { TypeId::of::<Enum>().field(2, 2).type_id() }, TypeId::of::<String>());
    /// ```
    ///
    /// The variant index and field index refer to the source order index of a variant in a type and
    /// the source order index of a field in a variant, respectively.
    ///
    /// For enums, variant indexes are always `0..variant_count`, regardless of any custom discriminants that may have been defined.
    /// `struct`s, `tuples`, and `unions`s are considered to have a single variant with variant index zero.
    ///
    /// As for field indexes, they may not be the same as the layout order for `repr(Rust)` types, but they are for `repr(C)` types.
    ///
    /// ```
    /// enum Enum {
    ///     Foo,  // variant index == 0
    ///     Bar { // variant index == 1
    ///         a: (), // field index == 0 in `Bar`
    ///         b: (), // field index == 1 in `Bar`
    ///     }
    /// }
    /// ```
    ///
    /// Out-of-bounds indexing will be treated as a compile-time error.
    ///
    /// ```compile_fail,E0080
    /// # #![feature(type_info)]
    /// # use std::any::TypeId;
    /// #
    /// # struct Point {
    /// #     x: u32,
    /// #     y: u32,
    /// # }
    /// # enum Enum {
    /// #     Unit,
    /// #     Tuple(u32, u64),
    /// #     Struct { x: u32, y: u32, z: String },
    /// # }
    /// const {
    ///     _ = TypeId::of::<Point>().field(0, 10); // error: indexing out of bounds: the len is 2 but the index is 10
    ///     _ = TypeId::of::<Enum>().field(2, 10); // error: indexing out of bounds: the len is 3 but the index is 10
    /// }
    /// ```
    #[unstable(feature = "type_info", issue = "146922")]
    #[rustc_const_unstable(feature = "type_info", issue = "146922")]
    #[rustc_comptime]
    pub fn field(self, variant_index: usize, field_index: usize) -> FieldId {
        FieldId {
            frt_type_id: intrinsics::type_id_field_representing_type(
                self,
                variant_index,
                field_index,
            ),
        }
    }
}

/// Field representing type ID. Representing a field of a struct, tuple or enum variant.
#[derive(Copy, PartialOrd, Ord, Hash)]
#[derive_const(Clone, PartialEq, Eq)]
#[unstable(feature = "type_info", issue = "146922")]
pub struct FieldId {
    frt_type_id: TypeId,
}

#[unstable(feature = "type_info", issue = "146922")]
impl fmt::Debug for FieldId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FieldId({:#034x})", self.frt_type_id.as_u128())
    }
}

impl FieldId {
    /// Returns the `TypeId` of the actual field type.
    ///
    /// ```
    /// #![feature(type_info)]
    /// use std::any::TypeId;
    ///
    /// struct Point {
    ///     x: u32,
    ///     y: u32,
    /// }
    /// assert_eq!(
    ///     const { TypeId::of::<Point>().field(0, 0).type_id() },
    ///     TypeId::of::<u32>()
    /// );
    /// ```
    #[unstable(feature = "type_info", issue = "146922")]
    #[rustc_const_unstable(feature = "type_info", issue = "146922")]
    #[rustc_comptime]
    pub fn type_id(self) -> TypeId {
        intrinsics::field_representing_type_actual_type_id(self.frt_type_id)
    }
}
