//! This module is used to store stuff from Rust's AST in a more convenient
//! manner (and with prettier names) before cleaning.
pub use self::StructType::*;

use syntax::ast;
use syntax::ast::{Name, NodeId};
use syntax::attr;
use syntax::ext::base::MacroKind;
use syntax_pos::{self, Span};

use rustc::hir;
use rustc::hir::def_id::CrateNum;
use rustc::hir::ptr::P;

pub struct Module<'hir> {
    pub name: Option<Name>,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub where_outer: Span,
    pub where_inner: Span,
    pub extern_crates: Vec<ExternCrate<'hir>>,
    pub imports: Vec<Import<'hir>>,
    pub structs: Vec<Struct<'hir>>,
    pub unions: Vec<Union<'hir>>,
    pub enums: Vec<Enum<'hir>>,
    pub fns: Vec<Function<'hir>>,
    pub mods: Vec<Module<'hir>>,
    pub id: NodeId,
    pub typedefs: Vec<Typedef<'hir>>,
    pub existentials: Vec<Existential<'hir>>,
    pub statics: Vec<Static<'hir>>,
    pub constants: Vec<Constant<'hir>>,
    pub traits: Vec<Trait<'hir>>,
    pub vis: &'hir hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
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
        attrs: &'hir hir::HirVec<ast::Attribute>,
        vis: &'hir hir::Visibility,
    ) -> Module<'hir> {
        Module {
            name       : name,
            id: ast::CRATE_NODE_ID,
            vis,
            stab: None,
            depr: None,
            where_outer: syntax_pos::DUMMY_SP,
            where_inner: syntax_pos::DUMMY_SP,
            attrs,
            extern_crates: Vec::new(),
            imports    :   Vec::new(),
            structs    :   Vec::new(),
            unions     :   Vec::new(),
            enums      :   Vec::new(),
            fns        :   Vec::new(),
            mods       :   Vec::new(),
            typedefs   :   Vec::new(),
            existentials:  Vec::new(),
            statics    :   Vec::new(),
            constants  :   Vec::new(),
            traits     :   Vec::new(),
            impls      :   Vec::new(),
            foreigns   :   Vec::new(),
            macros     :   Vec::new(),
            proc_macros:   Vec::new(),
            trait_aliases: Vec::new(),
            is_crate   : false,
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
    pub vis: &'hir hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: hir::HirId,
    pub struct_type: StructType,
    pub name: Name,
    pub generics: &'hir hir::Generics,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub fields: &'hir [hir::StructField],
    pub whence: Span,
}

pub struct Union<'hir> {
    pub vis: &'hir hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: hir::HirId,
    pub struct_type: StructType,
    pub name: Name,
    pub generics: &'hir hir::Generics,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub fields: &'hir [hir::StructField],
    pub whence: Span,
}

pub struct Enum<'hir> {
    pub vis: &'hir hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub variants: Vec<Variant<'hir>>,
    pub generics: &'hir hir::Generics,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub id: hir::HirId,
    pub whence: Span,
    pub name: Name,
}

pub struct Variant<'hir> {
    pub name: Name,
    pub id: hir::HirId,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub def: &'hir hir::VariantData,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub whence: Span,
}

pub struct Function<'hir> {
    pub decl: &'hir hir::FnDecl,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub id: hir::HirId,
    pub name: Name,
    pub vis: &'hir hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub header: hir::FnHeader,
    pub whence: Span,
    pub generics: &'hir hir::Generics,
    pub body: hir::BodyId,
}

pub struct Typedef<'hir> {
    pub ty: &'hir P<hir::Ty>,
    pub gen: &'hir hir::Generics,
    pub name: Name,
    pub id: hir::HirId,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub whence: Span,
    pub vis: &'hir hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
}

pub struct Existential<'hir> {
    pub exist_ty: &'hir hir::ExistTy,
    pub name: Name,
    pub id: hir::HirId,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub whence: Span,
    pub vis: &'hir hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
}

#[derive(Debug)]
pub struct Static<'hir> {
    pub type_: &'hir P<hir::Ty>,
    pub mutability: hir::Mutability,
    pub expr: hir::BodyId,
    pub name: Name,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub vis: &'hir hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: hir::HirId,
    pub whence: Span,
}

pub struct Constant<'hir> {
    pub type_: &'hir P<hir::Ty>,
    pub expr: hir::BodyId,
    pub name: Name,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub vis: &'hir hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: hir::HirId,
    pub whence: Span,
}

pub struct Trait<'hir> {
    pub is_auto: hir::IsAuto,
    pub unsafety: hir::Unsafety,
    pub name: Name,
    pub items: Vec<&'hir hir::TraitItem>,
    pub generics: &'hir hir::Generics,
    pub bounds: &'hir hir::HirVec<hir::GenericBound>,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub id: hir::HirId,
    pub whence: Span,
    pub vis: &'hir hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
}

pub struct TraitAlias<'hir> {
    pub name: Name,
    pub generics: &'hir hir::Generics,
    pub bounds: &'hir hir::HirVec<hir::GenericBound>,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub id: hir::HirId,
    pub whence: Span,
    pub vis: &'hir hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
}

#[derive(Debug)]
pub struct Impl<'hir> {
    pub unsafety: hir::Unsafety,
    pub polarity: hir::ImplPolarity,
    pub defaultness: hir::Defaultness,
    pub generics: &'hir hir::Generics,
    pub trait_: &'hir Option<hir::TraitRef>,
    pub for_: &'hir P<hir::Ty>,
    pub items: Vec<&'hir hir::ImplItem>,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub whence: Span,
    pub vis: &'hir hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: hir::HirId,
}

pub struct ForeignItem<'hir> {
    pub vis: &'hir hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: hir::HirId,
    pub name: Name,
    pub kind: &'hir hir::ForeignItemKind,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub whence: Span,
}

// For Macro we store the DefId instead of the NodeId, since we also create
// these imported macro_rules (which only have a DUMMY_NODE_ID).
pub struct Macro<'hir> {
    pub name: Name,
    pub def_id: hir::def_id::DefId,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub whence: Span,
    pub matchers: hir::HirVec<Span>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub imported_from: Option<Name>,
}

pub struct ExternCrate<'hir> {
    pub name: Name,
    pub cnum: CrateNum,
    pub path: Option<String>,
    pub vis: &'hir hir::Visibility,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub whence: Span,
}

pub struct Import<'hir> {
    pub name: Name,
    pub id: hir::HirId,
    pub vis: &'hir hir::Visibility,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub path: &'hir hir::Path,
    pub glob: bool,
    pub whence: Span,
}

pub struct ProcMacro<'hir> {
    pub name: Name,
    pub id: hir::HirId,
    pub kind: MacroKind,
    pub helpers: Vec<Name>,
    pub attrs: &'hir hir::HirVec<ast::Attribute>,
    pub whence: Span,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
}

pub fn struct_type_from_def(vdata: &hir::VariantData) -> StructType {
    match *vdata {
        hir::VariantData::Struct(..) => Plain,
        hir::VariantData::Tuple(..) => Tuple,
        hir::VariantData::Unit(..) => Unit,
    }
}
