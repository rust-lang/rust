//! This module is used to store stuff from Rust's AST in a more convenient
//! manner (and with prettier names) before cleaning.
crate use self::StructType::*;

use rustc_ast as ast;
use rustc_span::hygiene::MacroKind;
use rustc_span::{self, Span, Symbol};

use rustc_hir as hir;

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
    crate id: hir::HirId,
    crate struct_type: StructType,
    crate name: Symbol,
    crate generics: &'hir hir::Generics<'hir>,
    crate fields: &'hir [hir::StructField<'hir>],
}

crate struct Enum<'hir> {
    crate variants: Vec<Variant<'hir>>,
    crate generics: &'hir hir::Generics<'hir>,
    crate id: hir::HirId,
    crate name: Symbol,
}

crate struct Variant<'hir> {
    crate name: Symbol,
    crate id: hir::HirId,
    crate def: &'hir hir::VariantData<'hir>,
}

crate struct OpaqueTy<'hir> {
    crate opaque_ty: &'hir hir::OpaqueTy<'hir>,
    crate name: Symbol,
    crate id: hir::HirId,
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

crate struct Trait<'hir> {
    crate is_auto: hir::IsAuto,
    crate unsafety: hir::Unsafety,
    crate name: Symbol,
    crate items: Vec<&'hir hir::TraitItem<'hir>>,
    crate generics: &'hir hir::Generics<'hir>,
    crate bounds: &'hir [hir::GenericBound<'hir>],
    crate attrs: &'hir [ast::Attribute],
    crate id: hir::HirId,
}

crate struct TraitAlias<'hir> {
    crate name: Symbol,
    crate generics: &'hir hir::Generics<'hir>,
    crate bounds: &'hir [hir::GenericBound<'hir>],
    crate id: hir::HirId,
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

crate struct ProcMacro {
    crate name: Symbol,
    crate id: hir::HirId,
    crate kind: MacroKind,
    crate helpers: Vec<Symbol>,
}

crate fn struct_type_from_def(vdata: &hir::VariantData<'_>) -> StructType {
    match *vdata {
        hir::VariantData::Struct(..) => Plain,
        hir::VariantData::Tuple(..) => Tuple,
        hir::VariantData::Unit(..) => Unit,
    }
}
