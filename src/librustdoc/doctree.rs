//! This module is used to store stuff from Rust's AST in a more convenient
//! manner (and with prettier names) before cleaning.
pub use self::StructType::*;

use syntax::ast;
use syntax::ast::{Name, NodeId};
use syntax::attr;
use syntax::ext::base::MacroKind;
use syntax::source_map::Spanned;
use syntax_pos::{self, Span};

use rustc::hir;
use rustc::hir::def_id::CrateNum;
use rustc::hir::ptr::P;

pub struct Module<'a> {
    pub name: Option<Name>,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub where_outer: Span,
    pub where_inner: Span,
    pub extern_crates: Vec<ExternCrate<'a>>,
    pub imports: Vec<Import<'a>>,
    pub structs: Vec<Struct<'a>>,
    pub unions: Vec<Union<'a>>,
    pub enums: Vec<Enum<'a>>,
    pub fns: Vec<Function<'a>>,
    pub mods: Vec<Module<'a>>,
    pub id: NodeId,
    pub typedefs: Vec<Typedef<'a>>,
    pub existentials: Vec<Existential<'a>>,
    pub statics: Vec<Static<'a>>,
    pub constants: Vec<Constant<'a>>,
    pub traits: Vec<Trait<'a>>,
    pub vis: hir::Visibility<'a>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub impls: Vec<Impl<'a>>,
    pub foreigns: Vec<hir::ForeignMod<'a>>,
    pub macros: Vec<Macro<'a>>,
    pub proc_macros: Vec<ProcMacro<'a>>,
    pub trait_aliases: Vec<TraitAlias<'a>>,
    pub is_crate: bool,
}

impl Module<'_> {
    pub fn new(name: Option<Name>) -> Self {
        Module {
            name       : name,
            id: ast::CRATE_NODE_ID,
            vis: Spanned { span: syntax_pos::DUMMY_SP, node: hir::VisibilityKind::Inherited },
            stab: None,
            depr: None,
            where_outer: syntax_pos::DUMMY_SP,
            where_inner: syntax_pos::DUMMY_SP,
            attrs      : hir::HirVec::new(),
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

#[derive(Debug, Clone, RustcEncodable, RustcDecodable, Copy)]
pub enum StructType {
    /// A braced struct
    Plain,
    /// A tuple struct
    Tuple,
    /// A unit struct
    Unit,
}

pub struct Struct<'a> {
    pub vis: hir::Visibility<'a>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: hir::HirId,
    pub struct_type: StructType,
    pub name: Name,
    pub generics: hir::Generics<'a>,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub fields: Vec<hir::StructField<'a>>,
    pub whence: Span,
}

pub struct Union<'a> {
    pub vis: hir::Visibility<'a>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: hir::HirId,
    pub struct_type: StructType,
    pub name: Name,
    pub generics: hir::Generics<'a>,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub fields: Vec<hir::StructField<'a>>,
    pub whence: Span,
}

pub struct Enum<'a> {
    pub vis: hir::Visibility<'a>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub variants: Vec<Variant<'a>>,
    pub generics: hir::Generics<'a>,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub id: hir::HirId,
    pub whence: Span,
    pub name: Name,
}

pub struct Variant<'a> {
    pub name: Name,
    pub id: hir::HirId,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub def: hir::VariantData<'a>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub whence: Span,
}

pub struct Function<'a> {
    pub decl: hir::FnDecl<'a>,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub id: hir::HirId,
    pub name: Name,
    pub vis: hir::Visibility<'a>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub header: hir::FnHeader,
    pub whence: Span,
    pub generics: hir::Generics<'a>,
    pub body: hir::BodyId,
}

pub struct Typedef<'a> {
    pub ty: P<'a, hir::Ty<'a>>,
    pub gen: hir::Generics<'a>,
    pub name: Name,
    pub id: hir::HirId,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub whence: Span,
    pub vis: hir::Visibility<'a>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
}

pub struct Existential<'a> {
    pub exist_ty: hir::ExistTy<'a>,
    pub name: Name,
    pub id: hir::HirId,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub whence: Span,
    pub vis: hir::Visibility<'a>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
}

#[derive(Debug)]
pub struct Static<'a> {
    pub type_: P<'a, hir::Ty<'a>>,
    pub mutability: hir::Mutability,
    pub expr: hir::BodyId,
    pub name: Name,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub vis: hir::Visibility<'a>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: hir::HirId,
    pub whence: Span,
}

pub struct Constant<'a> {
    pub type_: P<'a, hir::Ty<'a>>,
    pub expr: hir::BodyId,
    pub name: Name,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub vis: hir::Visibility<'a>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: hir::HirId,
    pub whence: Span,
}

pub struct Trait<'a> {
    pub is_auto: hir::IsAuto,
    pub unsafety: hir::Unsafety,
    pub name: Name,
    pub items: Vec<hir::TraitItem<'a>>,
    pub generics: hir::Generics<'a>,
    pub bounds: Vec<hir::GenericBound<'a>>,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub id: hir::HirId,
    pub whence: Span,
    pub vis: hir::Visibility<'a>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
}

pub struct TraitAlias<'a> {
    pub name: Name,
    pub generics: hir::Generics<'a>,
    pub bounds: Vec<hir::GenericBound<'a>>,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub id: hir::HirId,
    pub whence: Span,
    pub vis: hir::Visibility<'a>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
}

#[derive(Debug)]
pub struct Impl<'a> {
    pub unsafety: hir::Unsafety,
    pub polarity: hir::ImplPolarity,
    pub defaultness: hir::Defaultness,
    pub generics: hir::Generics<'a>,
    pub trait_: Option<hir::TraitRef<'a>>,
    pub for_: P<'a, hir::Ty<'a>>,
    pub items: Vec<hir::ImplItem<'a>>,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub whence: Span,
    pub vis: hir::Visibility<'a>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub id: hir::HirId,
}

// For Macro we store the DefId instead of the NodeId, since we also create
// these imported macro_rules (which only have a DUMMY_NODE_ID).
pub struct Macro<'a> {
    pub name: Name,
    pub def_id: hir::def_id::DefId,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub whence: Span,
    pub matchers: Vec<Span>,
    pub stab: Option<attr::Stability>,
    pub depr: Option<attr::Deprecation>,
    pub imported_from: Option<Name>,
}

pub struct ExternCrate<'a> {
    pub name: Name,
    pub cnum: CrateNum,
    pub path: Option<String>,
    pub vis: hir::Visibility<'a>,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub whence: Span,
}

pub struct Import<'a> {
    pub name: Name,
    pub id: hir::HirId,
    pub vis: hir::Visibility<'a>,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
    pub path: hir::Path<'a>,
    pub glob: bool,
    pub whence: Span,
}

pub struct ProcMacro<'a> {
    pub name: Name,
    pub id: hir::HirId,
    pub kind: MacroKind,
    pub helpers: Vec<Name>,
    pub attrs: hir::HirVec<'a, ast::Attribute>,
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
