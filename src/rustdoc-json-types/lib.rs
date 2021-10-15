//! Rustdoc's JSON output interface
//!
//! These types are the public API exposed through the `--output-format json` flag. The [`Crate`]
//! struct is the root of the JSON blob and all other items are contained within.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// A `Crate` is the root of the emitted JSON blob. It contains all type/documentation information
/// about the language items in the local crate, as well as info about external items to allow
/// tools to find or link to them.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExternalCrate {
    pub name: String,
    pub html_root_url: Option<String>,
}

/// For external (not defined in the local crate) items, you don't get the same level of
/// information. This struct should contain enough to generate a link/reference to the item in
/// question, or can be used by a tool that takes the json output of multiple crates to find
/// the actual item definition with all the relevant info.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ItemSummary {
    /// Can be used to look up the name and html_root_url of the crate this item came from in the
    /// `external_crates` map.
    pub crate_id: u32,
    /// The list of path components for the fully qualified path of this item (e.g.
    /// `["std", "io", "lazy", "Lazy"]` for `std::io::lazy::Lazy`).
    pub path: Vec<String>,
    /// Whether this item is a struct, trait, macro, etc.
    pub kind: ItemKind,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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
    pub deprecation: Option<Deprecation>,
    #[serde(flatten)]
    pub inner: ItemEnum,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Span {
    /// The path to the source file for this span relative to the path `rustdoc` was invoked with.
    pub filename: PathBuf,
    /// Zero indexed Line and Column of the first character of the `Span`
    pub begin: (usize, usize),
    /// Zero indexed Line and Column of the last character of the `Span`
    pub end: (usize, usize),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Deprecation {
    pub since: Option<String>,
    pub note: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    Public,
    /// For the most part items are private by default. The exceptions are associated items of
    /// public traits and variants of public enums.
    Default,
    Crate,
    /// For `pub(in path)` visibility. `parent` is the module it's restricted to and `path` is how
    /// that module was referenced (like `"super::super"` or `"crate::foo::bar"`).
    Restricted {
        parent: Id,
        path: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GenericArgs {
    /// <'a, 32, B: Copy, C = u32>
    AngleBracketed { args: Vec<GenericArg>, bindings: Vec<TypeBinding> },
    /// Fn(A, B) -> C
    Parenthesized { inputs: Vec<Type>, output: Option<Type> },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GenericArg {
    Lifetime(String),
    Type(Type),
    Const(Constant),
    Infer,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Constant {
    #[serde(rename = "type")]
    pub type_: Type,
    pub expr: String,
    pub value: Option<String>,
    pub is_literal: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TypeBinding {
    pub name: String,
    pub binding: TypeBindingKind,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TypeBindingKind {
    Equality(Type),
    Constraint(Vec<GenericBound>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Id(pub String);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ItemKind {
    Module,
    ExternCrate,
    Import,
    Struct,
    StructField,
    Union,
    Enum,
    Variant,
    Function,
    Typedef,
    OpaqueTy,
    Constant,
    Trait,
    TraitAlias,
    Method,
    Impl,
    Static,
    ForeignType,
    Macro,
    ProcAttribute,
    ProcDerive,
    AssocConst,
    AssocType,
    Primitive,
    Keyword,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "inner", rename_all = "snake_case")]
pub enum ItemEnum {
    Module(Module),
    ExternCrate {
        name: String,
        rename: Option<String>,
    },
    Import(Import),

    Union(Union),
    Struct(Struct),
    StructField(Type),
    Enum(Enum),
    Variant(Variant),

    Function(Function),

    Trait(Trait),
    TraitAlias(TraitAlias),
    Method(Method),
    Impl(Impl),

    Typedef(Typedef),
    OpaqueTy(OpaqueTy),
    Constant(Constant),

    Static(Static),

    /// `type`s from an extern block
    ForeignType,

    /// Declarative macro_rules! macro
    Macro(String),
    ProcMacro(ProcMacro),

    PrimitiveType(String),

    AssocConst {
        #[serde(rename = "type")]
        type_: Type,
        /// e.g. `const X: usize = 5;`
        default: Option<String>,
    },
    AssocType {
        bounds: Vec<GenericBound>,
        /// e.g. `type X = usize;`
        default: Option<Type>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Module {
    pub is_crate: bool,
    pub items: Vec<Id>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Union {
    pub generics: Generics,
    pub fields_stripped: bool,
    pub fields: Vec<Id>,
    pub impls: Vec<Id>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Struct {
    pub struct_type: StructType,
    pub generics: Generics,
    pub fields_stripped: bool,
    pub fields: Vec<Id>,
    pub impls: Vec<Id>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Enum {
    pub generics: Generics,
    pub variants_stripped: bool,
    pub variants: Vec<Id>,
    pub impls: Vec<Id>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "variant_kind", content = "variant_inner")]
pub enum Variant {
    Plain,
    Tuple(Vec<Type>),
    Struct(Vec<Id>),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StructType {
    Plain,
    Tuple,
    Unit,
}

#[non_exhaustive]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Qualifiers {
    Const,
    Unsafe,
    Async,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Function {
    pub decl: FnDecl,
    pub generics: Generics,
    pub header: HashSet<Qualifiers>,
    pub abi: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Method {
    pub decl: FnDecl,
    pub generics: Generics,
    pub header: HashSet<Qualifiers>,
    pub abi: String,
    pub has_body: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Generics {
    pub params: Vec<GenericParamDef>,
    pub where_predicates: Vec<WherePredicate>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GenericParamDef {
    pub name: String,
    pub kind: GenericParamDefKind,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GenericParamDefKind {
    Lifetime { outlives: Vec<String> },
    Type { bounds: Vec<GenericBound>, default: Option<Type> },
    Const { ty: Type, default: Option<String> },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum WherePredicate {
    BoundPredicate { ty: Type, bounds: Vec<GenericBound> },
    RegionPredicate { lifetime: String, bounds: Vec<GenericBound> },
    EqPredicate { lhs: Type, rhs: Type },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GenericBound {
    TraitBound {
        #[serde(rename = "trait")]
        trait_: Type,
        /// Used for HRTBs
        generic_params: Vec<GenericParamDef>,
        modifier: TraitBoundModifier,
    },
    Outlives(String),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TraitBoundModifier {
    None,
    Maybe,
    MaybeConst,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "kind", content = "inner")]
pub enum Type {
    /// Structs, enums, and traits
    ResolvedPath {
        name: String,
        id: Id,
        args: Option<Box<GenericArgs>>,
        param_names: Vec<GenericBound>,
    },
    /// Parameterized types
    Generic(String),
    /// Fixed-size numeric types (plus int/usize/float), char, arrays, slices, and tuples
    Primitive(String),
    /// `extern "ABI" fn`
    FunctionPointer(Box<FunctionPointer>),
    /// `(String, u32, Box<usize>)`
    Tuple(Vec<Type>),
    /// `[u32]`
    Slice(Box<Type>),
    /// [u32; 15]
    Array {
        #[serde(rename = "type")]
        type_: Box<Type>,
        len: String,
    },
    /// `impl TraitA + TraitB + ...`
    ImplTrait(Vec<GenericBound>),
    /// `_`
    Infer,
    /// `*mut u32`, `*u8`, etc.
    RawPointer {
        mutable: bool,
        #[serde(rename = "type")]
        type_: Box<Type>,
    },
    /// `&'a mut String`, `&str`, etc.
    BorrowedRef {
        lifetime: Option<String>,
        mutable: bool,
        #[serde(rename = "type")]
        type_: Box<Type>,
    },
    /// `<Type as Trait>::Name` or associated types like `T::Item` where `T: Iterator`
    QualifiedPath {
        name: String,
        self_type: Box<Type>,
        #[serde(rename = "trait")]
        trait_: Box<Type>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FunctionPointer {
    pub decl: FnDecl,
    pub generic_params: Vec<GenericParamDef>,
    pub header: HashSet<Qualifiers>,
    pub abi: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FnDecl {
    pub inputs: Vec<(String, Type)>,
    pub output: Option<Type>,
    pub c_variadic: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Trait {
    pub is_auto: bool,
    pub is_unsafe: bool,
    pub items: Vec<Id>,
    pub generics: Generics,
    pub bounds: Vec<GenericBound>,
    pub implementors: Vec<Id>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TraitAlias {
    pub generics: Generics,
    pub params: Vec<GenericBound>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Impl {
    pub is_unsafe: bool,
    pub generics: Generics,
    pub provided_trait_methods: Vec<String>,
    #[serde(rename = "trait")]
    pub trait_: Option<Type>,
    #[serde(rename = "for")]
    pub for_: Type,
    pub items: Vec<Id>,
    pub negative: bool,
    pub synthetic: bool,
    pub blanket_impl: Option<Type>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct Import {
    /// The full path being imported.
    pub source: String,
    /// May be different from the last segment of `source` when renaming imports:
    /// `use source as name;`
    pub name: String,
    /// The ID of the item being imported.
    pub id: Option<Id>, // FIXME is this actually ever None?
    /// Whether this import uses a glob: `use source::*;`
    pub glob: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProcMacro {
    pub kind: MacroKind,
    pub helpers: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MacroKind {
    /// A bang macro `foo!()`.
    Bang,
    /// An attribute macro `#[foo]`.
    Attr,
    /// A derive macro `#[derive(Foo)]`
    Derive,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Typedef {
    #[serde(rename = "type")]
    pub type_: Type,
    pub generics: Generics,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OpaqueTy {
    pub bounds: Vec<GenericBound>,
    pub generics: Generics,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Static {
    #[serde(rename = "type")]
    pub type_: Type,
    pub mutable: bool,
    pub expr: String,
}

/// rustdoc format-version.
pub const FORMAT_VERSION: u32 = 9;

#[cfg(test)]
mod tests;
