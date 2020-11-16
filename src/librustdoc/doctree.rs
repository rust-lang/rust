//! This module is used to store stuff from Rust's AST in a more convenient
//! manner (and with prettier names) before cleaning.
crate use self::StructType::*;

use rustc_ast as ast;
use rustc_span::hygiene::MacroKind;
use rustc_span::{self, Span, Symbol};

use rustc_hir as hir;
use rustc_hir::def_id::CrateNum;
use rustc_hir::HirId;

crate struct Module<'hir> {
    crate name: Option<Symbol>,
    crate attrs: &'hir [ast::Attribute],
    crate where_outer: Span,
    crate where_inner: Span,
    crate extern_crates: Vec<ExternCrate<'hir>>,
    crate imports: Vec<Import<'hir>>,
    crate structs: Vec<Struct<'hir>>,
    crate unions: Vec<Union<'hir>>,
    crate enums: Vec<Enum<'hir>>,
    crate fns: Vec<Function<'hir>>,
    crate mods: Vec<Module<'hir>>,
    crate id: hir::HirId,
    crate typedefs: Vec<Typedef<'hir>>,
    crate opaque_tys: Vec<OpaqueTy<'hir>>,
    crate statics: Vec<Static<'hir>>,
    crate constants: Vec<Constant<'hir>>,
    crate traits: Vec<Trait<'hir>>,
    crate vis: &'hir hir::Visibility<'hir>,
    crate impls: Vec<Impl<'hir>>,
    crate foreigns: Vec<ForeignItem<'hir>>,
    crate macros: Vec<Macro<'hir>>,
    crate proc_macros: Vec<ProcMacro<'hir>>,
    crate trait_aliases: Vec<TraitAlias<'hir>>,
    crate is_crate: bool,
}

impl Module<'hir> {
    crate fn new(
        name: Option<Symbol>,
        attrs: &'hir [ast::Attribute],
        vis: &'hir hir::Visibility<'hir>,
    ) -> Module<'hir> {
        Module {
            name,
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
crate enum StructType {
    /// A braced struct
    Plain,
    /// A tuple struct
    Tuple,
    /// A unit struct
    Unit,
}

crate struct Struct<'hir> {
    crate vis: &'hir hir::Visibility<'hir>,
    crate id: hir::HirId,
    crate struct_type: StructType,
    crate name: Symbol,
    crate generics: &'hir hir::Generics<'hir>,
    crate attrs: &'hir [ast::Attribute],
    crate fields: &'hir [hir::StructField<'hir>],
    crate span: Span,
}

crate struct Union<'hir> {
    crate vis: &'hir hir::Visibility<'hir>,
    crate id: hir::HirId,
    crate struct_type: StructType,
    crate name: Symbol,
    crate generics: &'hir hir::Generics<'hir>,
    crate attrs: &'hir [ast::Attribute],
    crate fields: &'hir [hir::StructField<'hir>],
    crate span: Span,
}

crate struct Enum<'hir> {
    crate vis: &'hir hir::Visibility<'hir>,
    crate variants: Vec<Variant<'hir>>,
    crate generics: &'hir hir::Generics<'hir>,
    crate attrs: &'hir [ast::Attribute],
    crate id: hir::HirId,
    crate span: Span,
    crate name: Symbol,
}

crate struct Variant<'hir> {
    crate name: Symbol,
    crate id: hir::HirId,
    crate attrs: &'hir [ast::Attribute],
    crate def: &'hir hir::VariantData<'hir>,
    crate span: Span,
}

crate struct Function<'hir> {
    crate decl: &'hir hir::FnDecl<'hir>,
    crate attrs: &'hir [ast::Attribute],
    crate id: hir::HirId,
    crate name: Symbol,
    crate vis: &'hir hir::Visibility<'hir>,
    crate header: hir::FnHeader,
    crate span: Span,
    crate generics: &'hir hir::Generics<'hir>,
    crate body: hir::BodyId,
}

crate struct Typedef<'hir> {
    crate ty: &'hir hir::Ty<'hir>,
    crate gen: &'hir hir::Generics<'hir>,
    crate name: Symbol,
    crate id: hir::HirId,
    crate attrs: &'hir [ast::Attribute],
    crate span: Span,
    crate vis: &'hir hir::Visibility<'hir>,
}

crate struct OpaqueTy<'hir> {
    crate opaque_ty: &'hir hir::OpaqueTy<'hir>,
    crate name: Symbol,
    crate id: hir::HirId,
    crate attrs: &'hir [ast::Attribute],
    crate span: Span,
    crate vis: &'hir hir::Visibility<'hir>,
}

#[derive(Debug)]
crate struct Static<'hir> {
    crate type_: &'hir hir::Ty<'hir>,
    crate mutability: hir::Mutability,
    crate expr: hir::BodyId,
    crate name: Symbol,
    crate attrs: &'hir [ast::Attribute],
    crate vis: &'hir hir::Visibility<'hir>,
    crate id: hir::HirId,
    crate span: Span,
}

crate struct Constant<'hir> {
    crate type_: &'hir hir::Ty<'hir>,
    crate expr: hir::BodyId,
    crate name: Symbol,
    crate attrs: &'hir [ast::Attribute],
    crate vis: &'hir hir::Visibility<'hir>,
    crate id: hir::HirId,
    crate span: Span,
}

crate struct Trait<'hir> {
    crate is_auto: hir::IsAuto,
    crate unsafety: hir::Unsafety,
    crate name: Symbol,
    crate items: Vec<&'hir hir::TraitItem<'hir>>,
    crate generics: &'hir hir::Generics<'hir>,
    crate bounds: &'hir [hir::GenericBound<'hir>],
    crate attrs: &'hir [ast::Attribute],
    crate id: hir::HirId,
    crate span: Span,
    crate vis: &'hir hir::Visibility<'hir>,
}

crate struct TraitAlias<'hir> {
    crate name: Symbol,
    crate generics: &'hir hir::Generics<'hir>,
    crate bounds: &'hir [hir::GenericBound<'hir>],
    crate attrs: &'hir [ast::Attribute],
    crate id: hir::HirId,
    crate span: Span,
    crate vis: &'hir hir::Visibility<'hir>,
}

#[derive(Debug)]
crate struct Impl<'hir> {
    crate unsafety: hir::Unsafety,
    crate polarity: hir::ImplPolarity,
    crate defaultness: hir::Defaultness,
    crate constness: hir::Constness,
    crate generics: &'hir hir::Generics<'hir>,
    crate trait_: &'hir Option<hir::TraitRef<'hir>>,
    crate for_: &'hir hir::Ty<'hir>,
    crate items: Vec<&'hir hir::ImplItem<'hir>>,
    crate attrs: &'hir [ast::Attribute],
    crate span: Span,
    crate vis: &'hir hir::Visibility<'hir>,
    crate id: hir::HirId,
}

crate struct ForeignItem<'hir> {
    crate vis: &'hir hir::Visibility<'hir>,
    crate id: hir::HirId,
    crate name: Symbol,
    crate kind: &'hir hir::ForeignItemKind<'hir>,
    crate attrs: &'hir [ast::Attribute],
    crate span: Span,
}

// For Macro we store the DefId instead of the NodeId, since we also create
// these imported macro_rules (which only have a DUMMY_NODE_ID).
crate struct Macro<'hir> {
    crate name: Symbol,
    crate hid: hir::HirId,
    crate def_id: hir::def_id::DefId,
    crate attrs: &'hir [ast::Attribute],
    crate span: Span,
    crate matchers: Vec<Span>,
    crate imported_from: Option<Symbol>,
}

crate struct ExternCrate<'hir> {
    crate name: Symbol,
    crate hir_id: HirId,
    crate cnum: CrateNum,
    crate path: Option<String>,
    crate vis: &'hir hir::Visibility<'hir>,
    crate attrs: &'hir [ast::Attribute],
    crate span: Span,
}

#[derive(Debug)]
crate struct Import<'hir> {
    crate name: Symbol,
    crate id: hir::HirId,
    crate vis: &'hir hir::Visibility<'hir>,
    crate attrs: &'hir [ast::Attribute],
    crate path: &'hir hir::Path<'hir>,
    crate glob: bool,
    crate span: Span,
}

crate struct ProcMacro<'hir> {
    crate name: Symbol,
    crate id: hir::HirId,
    crate kind: MacroKind,
    crate helpers: Vec<Symbol>,
    crate attrs: &'hir [ast::Attribute],
    crate span: Span,
}

crate fn struct_type_from_def(vdata: &hir::VariantData<'_>) -> StructType {
    match *vdata {
        hir::VariantData::Struct(..) => Plain,
        hir::VariantData::Tuple(..) => Tuple,
        hir::VariantData::Unit(..) => Unit,
    }
}
