//! This module is used to store stuff from Rust's AST in a more convenient
//! manner (and with prettier names) before cleaning.
crate use self::StructType::*;

use rustc_ast as ast;
use rustc_span::{self, symbol::Ident, Span, Symbol};

use rustc_hir as hir;

crate struct Module<'hir> {
    crate name: Option<Symbol>,
    crate attrs: &'hir [ast::Attribute],
    crate where_outer: Span,
    crate where_inner: Span,
    crate imports: Vec<Import<'hir>>,
    crate mods: Vec<Module<'hir>>,
    crate id: hir::HirId,
    // (item, renamed)
    crate items: Vec<(&'hir hir::Item<'hir>, Option<Ident>)>,
    crate foreigns: Vec<(&'hir hir::ForeignItem<'hir>, Option<Ident>)>,
    crate macros: Vec<(&'hir hir::MacroDef<'hir>, Option<Ident>)>,
    crate is_crate: bool,
}

impl Module<'hir> {
    crate fn new(name: Option<Symbol>, attrs: &'hir [ast::Attribute]) -> Module<'hir> {
        Module {
            name,
            id: hir::CRATE_HIR_ID,
            where_outer: rustc_span::DUMMY_SP,
            where_inner: rustc_span::DUMMY_SP,
            attrs,
            imports: Vec::new(),
            mods: Vec::new(),
            items: Vec::new(),
            foreigns: Vec::new(),
            macros: Vec::new(),
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

crate struct Variant<'hir> {
    crate name: Symbol,
    crate id: hir::HirId,
    crate def: &'hir hir::VariantData<'hir>,
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

crate fn struct_type_from_def(vdata: &hir::VariantData<'_>) -> StructType {
    match *vdata {
        hir::VariantData::Struct(..) => Plain,
        hir::VariantData::Tuple(..) => Tuple,
        hir::VariantData::Unit(..) => Unit,
    }
}
