//! Rustdoc's JSON output interface
//!
//! These types are the public API exposed through the `--output-format json` flag. The [`Crate`]
//! struct is the root of the JSON blob and all other items are contained within.
//!
//! We expose a `rustc-hash` feature that is disabled by default. This feature switches the
//! [`std::collections::HashMap`] for [`rustc_hash::FxHashMap`] to improve the performance of said
//! `HashMap` in specific situations.
//!
//! `cargo-semver-checks` for example, saw a [-3% improvement][1] when benchmarking using the
//! `aws_sdk_ec2` JSON output (~500MB of JSON). As always, we recommend measuring the impact before
//! turning this feature on, as [`FxHashMap`][2] only concerns itself with hash speed, and may
//! increase the number of collisions.
//!
//! [1]: https://rust-lang.zulipchat.com/#narrow/channel/266220-t-rustdoc/topic/rustc-hash.20and.20performance.20of.20rustdoc-types/near/474855731
//! [2]: https://crates.io/crates/rustc-hash

#[cfg(not(feature = "rustc-hash"))]
use std::collections::HashMap;
use std::path::PathBuf;

#[cfg(feature = "rustc-hash")]
use rustc_hash::FxHashMap as HashMap;
use serde::{Deserialize, Serialize};

pub type FxHashMap<K, V> = HashMap<K, V>; // re-export for use in src/librustdoc

/// The version of JSON output that this crate represents.
///
/// This integer is incremented with every breaking change to the API,
/// and is returned along with the JSON blob as [`Crate::format_version`].
/// Consuming code should assert that this value matches the format version(s) that it supports.
pub const FORMAT_VERSION: u32 = 38;

/// The root of the emitted JSON blob.
///
/// It contains all type/documentation information
/// about the language items in the local crate, as well as info about external items to allow
/// tools to find or link to them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Crate {
    /// The id of the root [`Module`] item of the local crate.
    pub root: Id,
    /// The version string given to `--crate-version`, if any.
    pub crate_version: Option<String>,
    /// Whether or not the output includes private items.
    pub includes_private: bool,
    /// A collection of all items in the local crate as well as some external traits and their
    /// items that are referenced locally.
    pub index: HashMap<Id, Item>,
    /// Maps IDs to fully qualified paths and other info helpful for generating links.
    pub paths: HashMap<Id, ItemSummary>,
    /// Maps `crate_id` of items to a crate name and html_root_url if it exists.
    pub external_crates: HashMap<u32, ExternalCrate>,
    /// A single version number to be used in the future when making backwards incompatible changes
    /// to the JSON output.
    pub format_version: u32,
}

/// Metadata of a crate, either the same crate on which `rustdoc` was invoked, or its dependency.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExternalCrate {
    /// The name of the crate.
    pub name: String,
    /// The root URL at which the crate's documentation lives.
    pub html_root_url: Option<String>,
}

/// Information about an external (not defined in the local crate) [`Item`].
///
/// For external items, you don't get the same level of
/// information. This struct should contain enough to generate a link/reference to the item in
/// question, or can be used by a tool that takes the json output of multiple crates to find
/// the actual item definition with all the relevant info.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ItemSummary {
    /// Can be used to look up the name and html_root_url of the crate this item came from in the
    /// `external_crates` map.
    pub crate_id: u32,
    /// The list of path components for the fully qualified path of this item (e.g.
    /// `["std", "io", "lazy", "Lazy"]` for `std::io::lazy::Lazy`).
    ///
    /// Note that items can appear in multiple paths, and the one chosen is implementation
    /// defined. Currently, this is the full path to where the item was defined. Eg
    /// [`String`] is currently `["alloc", "string", "String"]` and [`HashMap`][`std::collections::HashMap`]
    /// is `["std", "collections", "hash", "map", "HashMap"]`, but this is subject to change.
    pub path: Vec<String>,
    /// Whether this item is a struct, trait, macro, etc.
    pub kind: ItemKind,
}

/// Anything that can hold documentation - modules, structs, enums, functions, traits, etc.
///
/// The `Item` data type holds fields that can apply to any of these,
/// and leaves kind-specific details (like function args or enum variants) to the `inner` field.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Item {
    /// The unique identifier of this item. Can be used to find this item in various mappings.
    pub id: Id,
    /// This can be used as a key to the `external_crates` map of [`Crate`] to see which crate
    /// this item came from.
    pub crate_id: u32,
    /// Some items such as impls don't have names.
    pub name: Option<String>,
    /// The source location of this item (absent if it came from a macro expansion or inline
    /// assembly).
    pub span: Option<Span>,
    /// By default all documented items are public, but you can tell rustdoc to output private items
    /// so this field is needed to differentiate.
    pub visibility: Visibility,
    /// The full markdown docstring of this item. Absent if there is no documentation at all,
    /// Some("") if there is some documentation but it is empty (EG `#[doc = ""]`).
    pub docs: Option<String>,
    /// This mapping resolves [intra-doc links](https://github.com/rust-lang/rfcs/blob/master/text/1946-intra-rustdoc-links.md) from the docstring to their IDs
    pub links: HashMap<String, Id>,
    /// Stringified versions of the attributes on this item (e.g. `"#[inline]"`)
    pub attrs: Vec<String>,
    /// Information about the item’s deprecation, if present.
    pub deprecation: Option<Deprecation>,
    /// The type-specific fields describing this item.
    pub inner: ItemEnum,
}

/// A range of source code.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    /// The path to the source file for this span relative to the path `rustdoc` was invoked with.
    pub filename: PathBuf,
    /// Zero indexed Line and Column of the first character of the `Span`
    pub begin: (usize, usize),
    /// Zero indexed Line and Column of the last character of the `Span`
    pub end: (usize, usize),
}

/// Information about the deprecation of an [`Item`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Deprecation {
    /// Usually a version number when this [`Item`] first became deprecated.
    pub since: Option<String>,
    /// The reason for deprecation and/or what alternatives to use.
    pub note: Option<String>,
}

/// Visibility of an [`Item`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    /// Explicitly public visibility set with `pub`.
    Public,
    /// For the most part items are private by default. The exceptions are associated items of
    /// public traits and variants of public enums.
    Default,
    /// Explicitly crate-wide visibility set with `pub(crate)`
    Crate,
    /// For `pub(in path)` visibility.
    Restricted {
        /// ID of the module to which this visibility restricts items.
        parent: Id,
        /// The path with which [`parent`] was referenced
        /// (like `super::super` or `crate::foo::bar`).
        ///
        /// [`parent`]: Visibility::Restricted::parent
        path: String,
    },
}

/// Dynamic trait object type (`dyn Trait`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DynTrait {
    /// All the traits implemented. One of them is the vtable, and the rest must be auto traits.
    pub traits: Vec<PolyTrait>,
    /// The lifetime of the whole dyn object
    /// ```text
    /// dyn Debug + 'static
    ///             ^^^^^^^
    ///             |
    ///             this part
    /// ```
    pub lifetime: Option<String>,
}

/// A trait and potential HRTBs
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PolyTrait {
    /// The path to the trait.
    #[serde(rename = "trait")]
    pub trait_: Path,
    /// Used for Higher-Rank Trait Bounds (HRTBs)
    /// ```text
    /// dyn for<'a> Fn() -> &'a i32"
    ///     ^^^^^^^
    /// ```
    pub generic_params: Vec<GenericParamDef>,
}

/// A set of generic arguments provided to a path segment, e.g.
///
/// ```text
/// std::option::Option::<u32>::None
///                      ^^^^^
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenericArgs {
    /// `<'a, 32, B: Copy, C = u32>`
    AngleBracketed {
        /// The list of each argument on this type.
        /// ```text
        /// <'a, 32, B: Copy, C = u32>
        ///  ^^^^^^
        /// ```
        args: Vec<GenericArg>,
        /// Associated type or constant bindings (e.g. `Item=i32` or `Item: Clone`) for this type.
        constraints: Vec<AssocItemConstraint>,
    },
    /// `Fn(A, B) -> C`
    Parenthesized {
        /// The input types, enclosed in parentheses.
        inputs: Vec<Type>,
        /// The output type provided after the `->`, if present.
        output: Option<Type>,
    },
}

/// One argument in a list of generic arguments to a path segment.
///
/// Part of [`GenericArgs`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenericArg {
    /// A lifetime argument.
    /// ```text
    /// std::borrow::Cow<'static, str>
    ///                  ^^^^^^^
    /// ```
    Lifetime(String),
    /// A type argument.
    /// ```text
    /// std::borrow::Cow<'static, str>
    ///                           ^^^
    /// ```
    Type(Type),
    /// A constant as a generic argument.
    /// ```text
    /// core::array::IntoIter<u32, { 640 * 1024 }>
    ///                            ^^^^^^^^^^^^^^
    /// ```
    Const(Constant),
    /// A generic argument that's explicitly set to be inferred.
    /// ```text
    /// std::vec::Vec::<_>::new()
    ///                 ^
    /// ```
    Infer,
}

/// A constant.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Constant {
    /// The stringified expression of this constant. Note that its mapping to the original
    /// source code is unstable and it's not guaranteed that it'll match the source code.
    pub expr: String,
    /// The value of the evaluated expression for this constant, which is only computed for numeric
    /// types.
    pub value: Option<String>,
    /// Whether this constant is a bool, numeric, string, or char literal.
    pub is_literal: bool,
}

/// Describes a bound applied to an associated type/constant.
///
/// Example:
/// ```text
/// IntoIterator<Item = u32, IntoIter: Clone>
///              ^^^^^^^^^^  ^^^^^^^^^^^^^^^
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssocItemConstraint {
    /// The name of the associated type/constant.
    pub name: String,
    /// Arguments provided to the associated type/constant.
    pub args: GenericArgs,
    /// The kind of bound applied to the associated type/constant.
    pub binding: AssocItemConstraintKind,
}

/// The way in which an associate type/constant is bound.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssocItemConstraintKind {
    /// The required value/type is specified exactly. e.g.
    /// ```text
    /// Iterator<Item = u32, IntoIter: DoubleEndedIterator>
    ///          ^^^^^^^^^^
    /// ```
    Equality(Term),
    /// The type is required to satisfy a set of bounds.
    /// ```text
    /// Iterator<Item = u32, IntoIter: DoubleEndedIterator>
    ///                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    /// ```
    Constraint(Vec<GenericBound>),
}

/// An opaque identifier for an item.
///
/// It can be used to lookup in [`Crate::index`] or [`Crate::paths`] to resolve it
/// to an [`Item`].
///
/// Id's are only valid within a single JSON blob. They cannot be used to
/// resolve references between the JSON output's for different crates.
///
/// Rustdoc makes no guarantees about the inner value of Id's. Applications
/// should treat them as opaque keys to lookup items, and avoid attempting
/// to parse them, or otherwise depend on any implementation details.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
// FIXME(aDotInTheVoid): Consider making this non-public in rustdoc-types.
pub struct Id(pub u32);

/// The fundamental kind of an item. Unlike [`ItemEnum`], this does not carry any additional info.
///
/// Part of [`ItemSummary`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemKind {
    /// A module declaration, e.g. `mod foo;` or `mod foo {}`
    Module,
    /// A crate imported via the `extern crate` syntax.
    ExternCrate,
    /// An import of 1 or more items into scope, using the `use` keyword.
    Use,
    /// A `struct` declaration.
    Struct,
    /// A field of a struct.
    StructField,
    /// A `union` declaration.
    Union,
    /// An `enum` declaration.
    Enum,
    /// A variant of a enum.
    Variant,
    /// A function declaration, e.g. `fn f() {}`
    Function,
    /// A type alias declaration, e.g. `type Pig = std::borrow::Cow<'static, str>;`
    TypeAlias,
    /// The declaration of a constant, e.g. `const GREETING: &str = "Hi :3";`
    Constant,
    /// A `trait` declaration.
    Trait,
    /// A trait alias declaration, e.g. `trait Int = Add + Sub + Mul + Div;`
    ///
    /// See [the tracking issue](https://github.com/rust-lang/rust/issues/41517)
    TraitAlias,
    /// An `impl` block.
    Impl,
    /// A `static` declaration.
    Static,
    /// `type`s from an `extern` block.
    ///
    /// See [the tracking issue](https://github.com/rust-lang/rust/issues/43467)
    ExternType,
    /// A macro declaration.
    ///
    /// Corresponds to either `ItemEnum::Macro(_)`
    /// or `ItemEnum::ProcMacro(ProcMacro { kind: MacroKind::Bang })`
    Macro,
    /// A procedural macro attribute.
    ///
    /// Corresponds to `ItemEnum::ProcMacro(ProcMacro { kind: MacroKind::Attr })`
    ProcAttribute,
    /// A procedural macro usable in the `#[derive()]` attribute.
    ///
    /// Corresponds to `ItemEnum::ProcMacro(ProcMacro { kind: MacroKind::Derive })`
    ProcDerive,
    /// An associated constant of a trait or a type.
    AssocConst,
    /// An associated type of a trait or a type.
    AssocType,
    /// A primitive type, e.g. `u32`.
    ///
    /// [`Item`]s of this kind only come from the core library.
    Primitive,
    /// A keyword declaration.
    ///
    /// [`Item`]s of this kind only come from the come library and exist solely
    /// to carry documentation for the respective keywords.
    Keyword,
}

/// Specific fields of an item.
///
/// Part of [`Item`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemEnum {
    /// A module declaration, e.g. `mod foo;` or `mod foo {}`
    Module(Module),
    /// A crate imported via the `extern crate` syntax.
    ExternCrate {
        /// The name of the imported crate.
        name: String,
        /// If the crate is renamed, this is its name in the crate.
        rename: Option<String>,
    },
    /// An import of 1 or more items into scope, using the `use` keyword.
    Use(Use),

    /// A `union` declaration.
    Union(Union),
    /// A `struct` declaration.
    Struct(Struct),
    /// A field of a struct.
    StructField(Type),
    /// An `enum` declaration.
    Enum(Enum),
    /// A variant of a enum.
    Variant(Variant),

    /// A function declaration (including methods and other associated functions)
    Function(Function),

    /// A `trait` declaration.
    Trait(Trait),
    /// A trait alias declaration, e.g. `trait Int = Add + Sub + Mul + Div;`
    ///
    /// See [the tracking issue](https://github.com/rust-lang/rust/issues/41517)
    TraitAlias(TraitAlias),
    /// An `impl` block.
    Impl(Impl),

    /// A type alias declaration, e.g. `type Pig = std::borrow::Cow<'static, str>;`
    TypeAlias(TypeAlias),
    /// The declaration of a constant, e.g. `const GREETING: &str = "Hi :3";`
    Constant {
        /// The type of the constant.
        #[serde(rename = "type")]
        type_: Type,
        /// The declared constant itself.
        #[serde(rename = "const")]
        const_: Constant,
    },

    /// A declaration of a `static`.
    Static(Static),

    /// `type`s from an `extern` block.
    ///
    /// See [the tracking issue](https://github.com/rust-lang/rust/issues/43467)
    ExternType,

    /// A macro_rules! declarative macro. Contains a single string with the source
    /// representation of the macro with the patterns stripped.
    Macro(String),
    /// A procedural macro.
    ProcMacro(ProcMacro),

    /// A primitive type, e.g. `u32`.
    ///
    /// [`Item`]s of this kind only come from the core library.
    Primitive(Primitive),

    /// An associated constant of a trait or a type.
    AssocConst {
        /// The type of the constant.
        #[serde(rename = "type")]
        type_: Type,
        /// Inside a trait declaration, this is the default value for the associated constant,
        /// if provided.
        /// Inside an `impl` block, this is the value assigned to the associated constant,
        /// and will always be present.
        ///
        /// The representation is implementation-defined and not guaranteed to be representative of
        /// either the resulting value or of the source code.
        ///
        /// ```rust
        /// const X: usize = 640 * 1024;
        /// //               ^^^^^^^^^^
        /// ```
        value: Option<String>,
    },
    /// An associated type of a trait or a type.
    AssocType {
        /// The generic parameters and where clauses on ahis associated type.
        generics: Generics,
        /// The bounds for this associated type. e.g.
        /// ```rust
        /// trait IntoIterator {
        ///     type Item;
        ///     type IntoIter: Iterator<Item = Self::Item>;
        /// //                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        /// }
        /// ```
        bounds: Vec<GenericBound>,
        /// Inside a trait declaration, this is the default for the associated type, if provided.
        /// Inside an impl block, this is the type assigned to the associated type, and will always
        /// be present.
        ///
        /// ```rust
        /// type X = usize;
        /// //       ^^^^^
        /// ```
        #[serde(rename = "type")]
        type_: Option<Type>,
    },
}

/// A module declaration, e.g. `mod foo;` or `mod foo {}`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Module {
    /// Whether this is the root item of a crate.
    ///
    /// This item doesn't correspond to any construction in the source code and is generated by the
    /// compiler.
    pub is_crate: bool,
    /// [`Item`]s declared inside this module.
    pub items: Vec<Id>,
    /// If `true`, this module is not part of the public API, but it contains
    /// items that are re-exported as public API.
    pub is_stripped: bool,
}

/// A `union`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Union {
    /// The generic parameters and where clauses on this union.
    pub generics: Generics,
    /// Whether any fields have been removed from the result, due to being private or hidden.
    pub has_stripped_fields: bool,
    /// The list of fields in the union.
    ///
    /// All of the corresponding [`Item`]s are of kind [`ItemEnum::StructField`].
    pub fields: Vec<Id>,
    /// All impls (both of traits and inherent) for this union.
    ///
    /// All of the corresponding [`Item`]s are of kind [`ItemEnum::Impl`].
    pub impls: Vec<Id>,
}

/// A `struct`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Struct {
    /// The kind of the struct (e.g. unit, tuple-like or struct-like) and the data specific to it,
    /// i.e. fields.
    pub kind: StructKind,
    /// The generic parameters and where clauses on this struct.
    pub generics: Generics,
    /// All impls (both of traits and inherent) for this struct.
    /// All of the corresponding [`Item`]s are of kind [`ItemEnum::Impl`].
    pub impls: Vec<Id>,
}

/// The kind of a [`Struct`] and the data specific to it, i.e. fields.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructKind {
    /// A struct with no fields and no parentheses.
    ///
    /// ```rust
    /// pub struct Unit;
    /// ```
    Unit,
    /// A struct with unnamed fields.
    ///
    /// All [`Id`]'s will point to [`ItemEnum::StructField`].
    /// Unlike most of JSON, private and `#[doc(hidden)]` fields will be given as `None`
    /// instead of being omitted, because order matters.
    ///
    /// ```rust
    /// pub struct TupleStruct(i32);
    /// pub struct EmptyTupleStruct();
    /// ```
    Tuple(Vec<Option<Id>>),
    /// A struct with named fields.
    ///
    /// ```rust
    /// pub struct PlainStruct { x: i32 }
    /// pub struct EmptyPlainStruct {}
    /// ```
    Plain {
        /// The list of fields in the struct.
        ///
        /// All of the corresponding [`Item`]s are of kind [`ItemEnum::StructField`].
        fields: Vec<Id>,
        /// Whether any fields have been removed from the result, due to being private or hidden.
        has_stripped_fields: bool,
    },
}

/// An `enum`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Enum {
    /// Information about the type parameters and `where` clauses of the enum.
    pub generics: Generics,
    /// Whether any variants have been removed from the result, due to being private or hidden.
    pub has_stripped_variants: bool,
    /// The list of variants in the enum.
    ///
    /// All of the corresponding [`Item`]s are of kind [`ItemEnum::Variant`]
    pub variants: Vec<Id>,
    /// `impl`s for the enum.
    pub impls: Vec<Id>,
}

/// A variant of an enum.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Variant {
    /// Whether the variant is plain, a tuple-like, or struct-like. Contains the fields.
    pub kind: VariantKind,
    /// The discriminant, if explicitly specified.
    pub discriminant: Option<Discriminant>,
}

/// The kind of an [`Enum`] [`Variant`] and the data specific to it, i.e. fields.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VariantKind {
    /// A variant with no parentheses
    ///
    /// ```rust
    /// enum Demo {
    ///     PlainVariant,
    ///     PlainWithDiscriminant = 1,
    /// }
    /// ```
    Plain,
    /// A variant with unnamed fields.
    ///
    /// All [`Id`]'s will point to [`ItemEnum::StructField`].
    /// Unlike most of JSON, `#[doc(hidden)]` fields will be given as `None`
    /// instead of being omitted, because order matters.
    ///
    /// ```rust
    /// enum Demo {
    ///     TupleVariant(i32),
    ///     EmptyTupleVariant(),
    /// }
    /// ```
    Tuple(Vec<Option<Id>>),
    /// A variant with named fields.
    ///
    /// ```rust
    /// enum Demo {
    ///     StructVariant { x: i32 },
    ///     EmptyStructVariant {},
    /// }
    /// ```
    Struct {
        /// The list of variants in the enum.
        /// All of the corresponding [`Item`]s are of kind [`ItemEnum::Variant`].
        fields: Vec<Id>,
        /// Whether any variants have been removed from the result, due to being private or hidden.
        has_stripped_fields: bool,
    },
}

/// The value that distinguishes a variant in an [`Enum`] from other variants.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Discriminant {
    /// The expression that produced the discriminant.
    ///
    /// Unlike `value`, this preserves the original formatting (eg suffixes,
    /// hexadecimal, and underscores), making it unsuitable to be machine
    /// interpreted.
    ///
    /// In some cases, when the value is too complex, this may be `"{ _ }"`.
    /// When this occurs is unstable, and may change without notice.
    pub expr: String,
    /// The numerical value of the discriminant. Stored as a string due to
    /// JSON's poor support for large integers, and the fact that it would need
    /// to store from [`i128::MIN`] to [`u128::MAX`].
    pub value: String,
}

/// A set of fundamental properties of a function.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FunctionHeader {
    /// Is this function marked as `const`?
    pub is_const: bool,
    /// Is this function unsafe?
    pub is_unsafe: bool,
    /// Is this function async?
    pub is_async: bool,
    /// The ABI used by the function.
    pub abi: Abi,
}

/// The ABI (Application Binary Interface) used by a function.
///
/// If a variant has an `unwind` field, this means the ABI that it represents can be specified in 2
/// ways: `extern "_"` and `extern "_-unwind"`, and a value of `true` for that field signifies the
/// latter variant.
///
/// See the [Rustonomicon section](https://doc.rust-lang.org/nightly/nomicon/ffi.html#ffi-and-unwinding)
/// on unwinding for more info.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Abi {
    // We only have a concrete listing here for stable ABI's because there are so many
    // See rustc_ast_passes::feature_gate::PostExpansionVisitor::check_abi for the list
    /// The default ABI, but that can also be written explicitly with `extern "Rust"`.
    Rust,
    /// Can be specified as `extern "C"` or, as a shorthand, just `extern`.
    C { unwind: bool },
    /// Can be specified as `extern "cdecl"`.
    Cdecl { unwind: bool },
    /// Can be specified as `extern "stdcall"`.
    Stdcall { unwind: bool },
    /// Can be specified as `extern "fastcall"`.
    Fastcall { unwind: bool },
    /// Can be specified as `extern "aapcs"`.
    Aapcs { unwind: bool },
    /// Can be specified as `extern "win64"`.
    Win64 { unwind: bool },
    /// Can be specified as `extern "sysv64"`.
    SysV64 { unwind: bool },
    /// Can be specified as `extern "system"`.
    System { unwind: bool },
    /// Any other ABI, including unstable ones.
    Other(String),
}

/// A function declaration (including methods and other associated functions).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Function {
    /// Information about the function signature, or declaration.
    pub sig: FunctionSignature,
    /// Information about the function’s type parameters and `where` clauses.
    pub generics: Generics,
    /// Information about core properties of the function, e.g. whether it's `const`, its ABI, etc.
    pub header: FunctionHeader,
    /// Whether the function has a body, i.e. an implementation.
    pub has_body: bool,
}

/// Generic parameters accepted by an item and `where` clauses imposed on it and the parameters.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Generics {
    /// A list of generic parameter definitions (e.g. `<T: Clone + Hash, U: Copy>`).
    pub params: Vec<GenericParamDef>,
    /// A list of where predicates (e.g. `where T: Iterator, T::Item: Copy`).
    pub where_predicates: Vec<WherePredicate>,
}

/// One generic parameter accepted by an item.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GenericParamDef {
    /// Name of the parameter.
    /// ```rust
    /// fn f<'resource, Resource>(x: &'resource Resource) {}
    /// //    ^^^^^^^^  ^^^^^^^^
    /// ```
    pub name: String,
    /// The kind of the parameter and data specific to a particular parameter kind, e.g. type
    /// bounds.
    pub kind: GenericParamDefKind,
}

/// The kind of a [`GenericParamDef`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenericParamDefKind {
    /// Denotes a lifetime parameter.
    Lifetime {
        /// Lifetimes that this lifetime parameter is required to outlive.
        ///
        /// ```rust
        /// fn f<'a, 'b, 'resource: 'a + 'b>(a: &'a str, b: &'b str, res: &'resource str) {}
        /// //                      ^^^^^^^
        /// ```
        outlives: Vec<String>,
    },

    /// Denotes a type parameter.
    Type {
        /// Bounds applied directly to the type. Note that the bounds from `where` clauses
        /// that constrain this parameter won't appear here.
        ///
        /// ```rust
        /// fn default2<T: Default>() -> [T; 2] where T: Clone { todo!() }
        /// //             ^^^^^^^
        /// ```
        bounds: Vec<GenericBound>,
        /// The default type for this parameter, if provided, e.g.
        ///
        /// ```rust
        /// trait PartialEq<Rhs = Self> {}
        /// //                    ^^^^
        /// ```
        default: Option<Type>,
        /// This is normally `false`, which means that this generic parameter is
        /// declared in the Rust source text.
        ///
        /// If it is `true`, this generic parameter has been introduced by the
        /// compiler behind the scenes.
        ///
        /// # Example
        ///
        /// Consider
        ///
        /// ```ignore (pseudo-rust)
        /// pub fn f(_: impl Trait) {}
        /// ```
        ///
        /// The compiler will transform this behind the scenes to
        ///
        /// ```ignore (pseudo-rust)
        /// pub fn f<impl Trait: Trait>(_: impl Trait) {}
        /// ```
        ///
        /// In this example, the generic parameter named `impl Trait` (and which
        /// is bound by `Trait`) is synthetic, because it was not originally in
        /// the Rust source text.
        is_synthetic: bool,
    },

    /// Denotes a constant parameter.
    Const {
        /// The type of the constant as declared.
        #[serde(rename = "type")]
        type_: Type,
        /// The stringified expression for the default value, if provided. It's not guaranteed that
        /// it'll match the actual source code for the default value.
        default: Option<String>,
    },
}

/// One `where` clause.
/// ```rust
/// fn default<T>() -> T where T: Default { T::default() }
/// //                         ^^^^^^^^^^
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WherePredicate {
    /// A type is expected to comply with a set of bounds
    BoundPredicate {
        /// The type that's being constrained.
        ///
        /// ```rust
        /// fn f<T>(x: T) where for<'a> &'a T: Iterator {}
        /// //                              ^
        /// ```
        #[serde(rename = "type")]
        type_: Type,
        /// The set of bounds that constrain the type.
        ///
        /// ```rust
        /// fn f<T>(x: T) where for<'a> &'a T: Iterator {}
        /// //                                 ^^^^^^^^
        /// ```
        bounds: Vec<GenericBound>,
        /// Used for Higher-Rank Trait Bounds (HRTBs)
        /// ```rust
        /// fn f<T>(x: T) where for<'a> &'a T: Iterator {}
        /// //                  ^^^^^^^
        /// ```
        generic_params: Vec<GenericParamDef>,
    },

    /// A lifetime is expected to outlive other lifetimes.
    LifetimePredicate {
        /// The name of the lifetime.
        lifetime: String,
        /// The lifetimes that must be encompassed by the lifetime.
        outlives: Vec<String>,
    },

    /// A type must exactly equal another type.
    EqPredicate {
        /// The left side of the equation.
        lhs: Type,
        /// The right side of the equation.
        rhs: Term,
    },
}

/// Either a trait bound or a lifetime bound.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenericBound {
    /// A trait bound.
    TraitBound {
        /// The full path to the trait.
        #[serde(rename = "trait")]
        trait_: Path,
        /// Used for Higher-Rank Trait Bounds (HRTBs)
        /// ```text
        /// where F: for<'a, 'b> Fn(&'a u8, &'b u8)
        ///          ^^^^^^^^^^^
        ///          |
        ///          this part
        /// ```
        generic_params: Vec<GenericParamDef>,
        /// The context for which a trait is supposed to be used, e.g. `const
        modifier: TraitBoundModifier,
    },
    /// A lifetime bound, e.g.
    /// ```rust
    /// fn f<'a, T>(x: &'a str, y: &T) where T: 'a {}
    /// //                                     ^^^
    /// ```
    Outlives(String),
    /// `use<'a, T>` precise-capturing bound syntax
    Use(Vec<String>),
}

/// A set of modifiers applied to a trait.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraitBoundModifier {
    /// Marks the absence of a modifier.
    None,
    /// Indicates that the trait bound relaxes a trait bound applied to a parameter by default,
    /// e.g. `T: Sized?`, the `Sized` trait is required for all generic type parameters by default
    /// unless specified otherwise with this modifier.
    Maybe,
    /// Indicates that the trait bound must be applicable in both a run-time and a compile-time
    /// context.
    MaybeConst,
}

/// Either a type or a constant, usually stored as the right-hand side of an equation in places like
/// [`AssocItemConstraint`]
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Term {
    /// A type.
    ///
    /// ```rust
    /// fn f(x: impl IntoIterator<Item = u32>) {}
    /// //                               ^^^
    /// ```
    Type(Type),
    /// A constant.
    ///
    /// ```ignore (incomplete feature in the snippet)
    /// trait Foo {
    ///     const BAR: usize;
    /// }
    ///
    /// fn f(x: impl Foo<BAR = 42>) {}
    /// //                     ^^
    /// ```
    Constant(Constant),
}

/// A type.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Type {
    /// Structs, enums, unions and type aliases, e.g. `std::option::Option<u32>`
    ResolvedPath(Path),
    /// Dynamic trait object type (`dyn Trait`).
    DynTrait(DynTrait),
    /// Parameterized types. The contained string is the name of the parameter.
    Generic(String),
    /// Built-in numeric types (e.g. `u32`, `f32`), `bool`, `char`.
    Primitive(String),
    /// A function pointer type, e.g. `fn(u32) -> u32`, `extern "C" fn() -> *const u8`
    FunctionPointer(Box<FunctionPointer>),
    /// A tuple type, e.g. `(String, u32, Box<usize>)`
    Tuple(Vec<Type>),
    /// An unsized slice type, e.g. `[u32]`.
    Slice(Box<Type>),
    /// An array type, e.g. `[u32; 15]`
    Array {
        /// The type of the contained element.
        #[serde(rename = "type")]
        type_: Box<Type>,
        /// The stringified expression that is the length of the array.
        ///
        /// Keep in mind that it's not guaranteed to match the actual source code of the expression.
        len: String,
    },
    /// A pattern type, e.g. `u32 is 1..`
    ///
    /// See [the tracking issue](https://github.com/rust-lang/rust/issues/123646)
    Pat {
        /// The base type, e.g. the `u32` in `u32 is 1..`
        #[serde(rename = "type")]
        type_: Box<Type>,
        #[doc(hidden)]
        __pat_unstable_do_not_use: String,
    },
    /// An opaque type that satisfies a set of bounds, `impl TraitA + TraitB + ...`
    ImplTrait(Vec<GenericBound>),
    /// A type that's left to be inferred, `_`
    Infer,
    /// A raw pointer type, e.g. `*mut u32`, `*const u8`, etc.
    RawPointer {
        /// This is `true` for `*mut _` and `false` for `*const _`.
        is_mutable: bool,
        /// The type of the pointee.
        #[serde(rename = "type")]
        type_: Box<Type>,
    },
    /// `&'a mut String`, `&str`, etc.
    BorrowedRef {
        /// The name of the lifetime of the reference, if provided.
        lifetime: Option<String>,
        /// This is `true` for `&mut i32` and `false` for `&i32`
        is_mutable: bool,
        /// The type of the pointee, e.g. the `i32` in `&'a mut i32`
        #[serde(rename = "type")]
        type_: Box<Type>,
    },
    /// Associated types like `<Type as Trait>::Name` and `T::Item` where
    /// `T: Iterator` or inherent associated types like `Struct::Name`.
    QualifiedPath {
        /// The name of the associated type in the parent type.
        ///
        /// ```ignore (incomplete expression)
        /// <core::array::IntoIter<u32, 42> as Iterator>::Item
        /// //                                            ^^^^
        /// ```
        name: String,
        /// The generic arguments provided to the associated type.
        ///
        /// ```ignore (incomplete expression)
        /// <core::slice::IterMut<'static, u32> as BetterIterator>::Item<'static>
        /// //                                                          ^^^^^^^^^
        /// ```
        args: Box<GenericArgs>,
        /// The type with which this type is associated.
        ///
        /// ```ignore (incomplete expression)
        /// <core::array::IntoIter<u32, 42> as Iterator>::Item
        /// // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        /// ```
        self_type: Box<Type>,
        /// `None` iff this is an *inherent* associated type.
        #[serde(rename = "trait")]
        trait_: Option<Path>,
    },
}

/// A type that has a simple path to it. This is the kind of type of structs, unions, enums, etc.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Path {
    /// The name of the type as declared, e.g. in
    ///
    /// ```rust
    /// mod foo {
    ///     struct Bar;
    /// }
    /// ```
    ///
    /// for `foo::Bar`, this field will be `Bar`.
    pub name: String,
    /// The ID of the type.
    pub id: Id,
    /// Generic arguments to the type.
    ///
    /// ```ignore (incomplete expression)
    /// std::borrow::Cow<'static, str>
    /// //              ^^^^^^^^^^^^^^
    /// ```
    pub args: Option<Box<GenericArgs>>,
}

/// A type that is a function pointer.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FunctionPointer {
    /// The signature of the function.
    pub sig: FunctionSignature,
    /// Used for Higher-Rank Trait Bounds (HRTBs)
    ///
    /// ```ignore (incomplete expression)
    ///    for<'c> fn(val: &'c i32) -> i32
    /// // ^^^^^^^
    /// ```
    pub generic_params: Vec<GenericParamDef>,
    /// The core properties of the function, such as the ABI it conforms to, whether it's unsafe, etc.
    pub header: FunctionHeader,
}

/// The signature of a function.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FunctionSignature {
    /// List of argument names and their type.
    ///
    /// Note that not all names will be valid identifiers, as some of
    /// them may be patterns.
    pub inputs: Vec<(String, Type)>,
    /// The output type, if specified.
    pub output: Option<Type>,
    /// Whether the function accepts an arbitrary amount of trailing arguments the C way.
    ///
    /// ```ignore (incomplete code)
    /// fn printf(fmt: &str, ...);
    /// ```
    pub is_c_variadic: bool,
}

/// A `trait` declaration.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Trait {
    /// Whether the trait is marked `auto` and is thus implemented automatically
    /// for all applicable types.
    pub is_auto: bool,
    /// Whether the trait is marked as `unsafe`.
    pub is_unsafe: bool,
    // FIXME(dyn_compat_renaming): Update the URL once the Reference is updated and hits stable.
    /// Whether the trait is [dyn compatible](https://doc.rust-lang.org/reference/items/traits.html#object-safety)[^1].
    ///
    /// [^1]: Formerly known as "object safe".
    pub is_dyn_compatible: bool,
    /// Associated [`Item`]s that can/must be implemented by the `impl` blocks.
    pub items: Vec<Id>,
    /// Information about the type parameters and `where` clauses of the trait.
    pub generics: Generics,
    /// Constraints that must be met by the implementor of the trait.
    pub bounds: Vec<GenericBound>,
    /// The implementations of the trait.
    pub implementations: Vec<Id>,
}

/// A trait alias declaration, e.g. `trait Int = Add + Sub + Mul + Div;`
///
/// See [the tracking issue](https://github.com/rust-lang/rust/issues/41517)
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TraitAlias {
    /// Information about the type parameters and `where` clauses of the alias.
    pub generics: Generics,
    /// The bounds that are associated with the alias.
    pub params: Vec<GenericBound>,
}

/// An `impl` block.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Impl {
    /// Whether this impl is for an unsafe trait.
    pub is_unsafe: bool,
    /// Information about the impl’s type parameters and `where` clauses.
    pub generics: Generics,
    /// The list of the names of all the trait methods that weren't mentioned in this impl but
    /// were provided by the trait itself.
    ///
    /// For example, for this impl of the [`PartialEq`] trait:
    /// ```rust
    /// struct Foo;
    ///
    /// impl PartialEq for Foo {
    ///     fn eq(&self, other: &Self) -> bool { todo!() }
    /// }
    /// ```
    /// This field will be `["ne"]`, as it has a default implementation defined for it.
    pub provided_trait_methods: Vec<String>,
    /// The trait being implemented or `None` if the impl is inherent, which means
    /// `impl Struct {}` as opposed to `impl Trait for Struct {}`.
    #[serde(rename = "trait")]
    pub trait_: Option<Path>,
    /// The type that the impl block is for.
    #[serde(rename = "for")]
    pub for_: Type,
    /// The list of associated items contained in this impl block.
    pub items: Vec<Id>,
    /// Whether this is a negative impl (e.g. `!Sized` or `!Send`).
    pub is_negative: bool,
    /// Whether this is an impl that’s implied by the compiler
    /// (for autotraits, e.g. `Send` or `Sync`).
    pub is_synthetic: bool,
    // FIXME: document this
    pub blanket_impl: Option<Type>,
}

/// A `use` statement.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Use {
    /// The full path being imported.
    pub source: String,
    /// May be different from the last segment of `source` when renaming imports:
    /// `use source as name;`
    pub name: String,
    /// The ID of the item being imported. Will be `None` in case of re-exports of primitives:
    /// ```rust
    /// pub use i32 as my_i32;
    /// ```
    pub id: Option<Id>,
    /// Whether this statement is a wildcard `use`, e.g. `use source::*;`
    pub is_glob: bool,
}

/// A procedural macro.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProcMacro {
    /// How this macro is supposed to be called: `foo!()`, `#[foo]` or `#[derive(foo)]`
    pub kind: MacroKind,
    /// Helper attributes defined by a macro to be used inside it.
    ///
    /// Defined only for derive macros.
    ///
    /// E.g. the [`Default`] derive macro defines a `#[default]` helper attribute so that one can
    /// do:
    ///
    /// ```rust
    /// #[derive(Default)]
    /// enum Option<T> {
    ///     #[default]
    ///     None,
    ///     Some(T),
    /// }
    /// ```
    pub helpers: Vec<String>,
}

/// The way a [`ProcMacro`] is declared to be used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MacroKind {
    /// A bang macro `foo!()`.
    Bang,
    /// An attribute macro `#[foo]`.
    Attr,
    /// A derive macro `#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]`
    Derive,
}

/// A type alias declaration, e.g. `type Pig = std::borrow::Cow<'static, str>;`
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeAlias {
    /// The type referred to by this alias.
    #[serde(rename = "type")]
    pub type_: Type,
    /// Information about the type parameters and `where` clauses of the alias.
    pub generics: Generics,
}

/// A `static` declaration.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Static {
    /// The type of the static.
    #[serde(rename = "type")]
    pub type_: Type,
    /// This is `true` for mutable statics, declared as `static mut X: T = f();`
    pub is_mutable: bool,
    /// The stringified expression for the initial value.
    ///
    /// It's not guaranteed that it'll match the actual source code for the initial value.
    pub expr: String,

    /// Is the static `unsafe`?
    ///
    /// This is only true if it's in an `extern` block, and not explicity marked
    /// as `safe`.
    ///
    /// ```rust
    /// unsafe extern {
    ///     static A: i32;      // unsafe
    ///     safe static B: i32; // safe
    /// }
    ///
    /// static C: i32 = 0;     // safe
    /// static mut D: i32 = 0; // safe
    /// ```
    pub is_unsafe: bool,
}

/// A primitive type declaration. Declarations of this kind can only come from the core library.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Primitive {
    /// The name of the type.
    pub name: String,
    /// The implementations, inherent and of traits, on the primitive type.
    pub impls: Vec<Id>,
}

#[cfg(test)]
mod tests;
