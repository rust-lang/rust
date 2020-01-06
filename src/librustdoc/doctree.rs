//! This module is used to store stuff from Rust's AST in a more convenient
//! manner (and with prettier names) before cleaning.
pub use self::StructType::*;

use rustc_span::hygiene::MacroKind;
use rustc_span::{self, Span};
use syntax::ast;
use syntax::ast::Name;

use rustc_hir as hir;
use rustc_hir::def_id::CrateNum;

pub struct Module<'hir> {
    pub name: Option<Name>,
    pub attrs: &'hir [ast::Attribute],
    pub where_outer: Span,
    pub where_inner: Span,
    pub extern_crates: Vec<ExternCrate<'hir>>,
    pub imports: Vec<Import<'hir>>,
    pub structs: Vec<Struct<'hir>>,
    pub unions: Vec<Union<'hir>>,
    pub enums: Vec<Enum<'hir>>,
    pub fns: Vec<Function<'hir>>,
    pub mods: Vec<Module<'hir>>,
    pub id: hir::HirId,
    pub typedefs: Vec<Typedef<'hir>>,
    pub opaque_tys: Vec<OpaqueTy<'hir>>,
    pub statics: Vec<Static<'hir>>,
    pub constants: Vec<Constant<'hir>>,
    pub traits: Vec<Trait<'hir>>,
    pub vis: &'hir hir::Visibility<'hir>,
    pub impls: Vec<Impl<'hir>>,
    pub foreigns: Vec<ForeignItem<'hir>>,
    pub macros: Vec<Macro<'hir>>,
    pub proc_macros: Vec<ProcMacro<'hir>>,
    pub trait_aliases: Vec<TraitAlias<'hir>>,
    pub is_crate: bool,
}

impl Module<'hir> {
    pub fn new(
        name: Option<Name>,
        attrs: &'hir [ast::Attribute],
        vis: &'hir hir::Visibility<'hir>,
    ) -> Module<'hir> {
        Module {
            name: name,
            id: hir::CRATE_HIR_ID,
            vis,
            where_outer: rustc_span::DUMMY_SP,
            where_inner: rustc_span::DUMMY_SP,
            attrs,
            extern_crates: Vec::new(),
            imports: Vec::new(),
            structs: Vec::new(),
            unions: Vec::new(),
            enums: Vec::new(),
            fns: Vec::new(),
            mods: Vec::new(),
            typedefs: Vec::new(),
            opaque_tys: Vec::new(),
            statics: Vec::new(),
            constants: Vec::new(),
            traits: Vec::new(),
            impls: Vec::new(),
            foreigns: Vec::new(),
            macros: Vec::new(),
            proc_macros: Vec::new(),
            trait_aliases: Vec::new(),
            is_crate: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum StructType {
    /// A braced struct
    Plain,
    /// A tuple struct
    Tuple,
    /// A unit struct
    Unit,
}

pub struct Struct<'hir> {
    pub vis: &'hir hir::Visibility<'hir>,
    pub id: hir::HirId,
    pub struct_type: StructType,
    pub name: Name,
    pub generics: &'hir hir::Generics<'hir>,
    pub attrs: &'hir [ast::Attribute],
    pub fields: &'hir [hir::StructField<'hir>],
    pub whence: Span,
}

pub struct Union<'hir> {
    pub vis: &'hir hir::Visibility<'hir>,
    pub id: hir::HirId,
    pub struct_type: StructType,
    pub name: Name,
    pub generics: &'hir hir::Generics<'hir>,
    pub attrs: &'hir [ast::Attribute],
    pub fields: &'hir [hir::StructField<'hir>],
    pub whence: Span,
}

pub struct Enum<'hir> {
    pub vis: &'hir hir::Visibility<'hir>,
    pub variants: Vec<Variant<'hir>>,
    pub generics: &'hir hir::Generics<'hir>,
    pub attrs: &'hir [ast::Attribute],
    pub id: hir::HirId,
    pub whence: Span,
    pub name: Name,
}

pub struct Variant<'hir> {
    pub name: Name,
    pub id: hir::HirId,
    pub attrs: &'hir [ast::Attribute],
    pub def: &'hir hir::VariantData<'hir>,
    pub whence: Span,
}

pub struct Function<'hir> {
    pub decl: &'hir hir::FnDecl<'hir>,
    pub attrs: &'hir [ast::Attribute],
    pub id: hir::HirId,
    pub name: Name,
    pub vis: &'hir hir::Visibility<'hir>,
    pub header: hir::FnHeader,
    pub whence: Span,
    pub generics: &'hir hir::Generics<'hir>,
    pub body: hir::BodyId,
}

pub struct Typedef<'hir> {
    pub ty: &'hir hir::Ty<'hir>,
    pub gen: &'hir hir::Generics<'hir>,
    pub name: Name,
    pub id: hir::HirId,
    pub attrs: &'hir [ast::Attribute],
    pub whence: Span,
    pub vis: &'hir hir::Visibility<'hir>,
}

pub struct OpaqueTy<'hir> {
    pub opaque_ty: &'hir hir::OpaqueTy<'hir>,
    pub name: Name,
    pub id: hir::HirId,
    pub attrs: &'hir [ast::Attribute],
    pub whence: Span,
    pub vis: &'hir hir::Visibility<'hir>,
}

#[derive(Debug)]
pub struct Static<'hir> {
    pub type_: &'hir hir::Ty<'hir>,
    pub mutability: hir::Mutability,
    pub expr: hir::BodyId,
    pub name: Name,
    pub attrs: &'hir [ast::Attribute],
    pub vis: &'hir hir::Visibility<'hir>,
    pub id: hir::HirId,
    pub whence: Span,
}

pub struct Constant<'hir> {
    pub type_: &'hir hir::Ty<'hir>,
    pub expr: hir::BodyId,
    pub name: Name,
    pub attrs: &'hir [ast::Attribute],
    pub vis: &'hir hir::Visibility<'hir>,
    pub id: hir::HirId,
    pub whence: Span,
}

pub struct Trait<'hir> {
    pub is_auto: hir::IsAuto,
    pub unsafety: hir::Unsafety,
    pub name: Name,
    pub items: Vec<&'hir hir::TraitItem<'hir>>,
    pub generics: &'hir hir::Generics<'hir>,
    pub bounds: &'hir [hir::GenericBound<'hir>],
    pub attrs: &'hir [ast::Attribute],
    pub id: hir::HirId,
    pub whence: Span,
    pub vis: &'hir hir::Visibility<'hir>,
}

pub struct TraitAlias<'hir> {
    pub name: Name,
    pub generics: &'hir hir::Generics<'hir>,
    pub bounds: &'hir [hir::GenericBound<'hir>],
    pub attrs: &'hir [ast::Attribute],
    pub id: hir::HirId,
    pub whence: Span,
    pub vis: &'hir hir::Visibility<'hir>,
}

#[derive(Debug)]
pub struct Impl<'hir> {
    pub unsafety: hir::Unsafety,
    pub polarity: hir::ImplPolarity,
    pub defaultness: hir::Defaultness,
    pub generics: &'hir hir::Generics<'hir>,
    pub trait_: &'hir Option<hir::TraitRef<'hir>>,
    pub for_: &'hir hir::Ty<'hir>,
    pub items: Vec<&'hir hir::ImplItem<'hir>>,
    pub attrs: &'hir [ast::Attribute],
    pub whence: Span,
    pub vis: &'hir hir::Visibility<'hir>,
    pub id: hir::HirId,
}

pub struct ForeignItem<'hir> {
    pub vis: &'hir hir::Visibility<'hir>,
    pub id: hir::HirId,
    pub name: Name,
    pub kind: &'hir hir::ForeignItemKind<'hir>,
    pub attrs: &'hir [ast::Attribute],
    pub whence: Span,
}

// For Macro we store the DefId instead of the NodeId, since we also create
// these imported macro_rules (which only have a DUMMY_NODE_ID).
pub struct Macro<'hir> {
    pub name: Name,
    pub hid: hir::HirId,
    pub def_id: hir::def_id::DefId,
    pub attrs: &'hir [ast::Attribute],
    pub whence: Span,
    pub matchers: Vec<Span>,
    pub imported_from: Option<Name>,
}

pub struct ExternCrate<'hir> {
    pub name: Name,
    pub cnum: CrateNum,
    pub path: Option<String>,
    pub vis: &'hir hir::Visibility<'hir>,
    pub attrs: &'hir [ast::Attribute],
    pub whence: Span,
}

pub struct Import<'hir> {
    pub name: Name,
    pub id: hir::HirId,
    pub vis: &'hir hir::Visibility<'hir>,
    pub attrs: &'hir [ast::Attribute],
    pub path: &'hir hir::Path<'hir>,
    pub glob: bool,
    pub whence: Span,
}

pub struct ProcMacro<'hir> {
    pub name: Name,
    pub id: hir::HirId,
    pub kind: MacroKind,
    pub helpers: Vec<Name>,
    pub attrs: &'hir [ast::Attribute],
    pub whence: Span,
}

pub fn struct_type_from_def(vdata: &hir::VariantData) -> StructType {
    match *vdata {
        hir::VariantData::Struct(..) => Plain,
        hir::VariantData::Tuple(..) => Tuple,
        hir::VariantData::Unit(..) => Unit,
    }
}
