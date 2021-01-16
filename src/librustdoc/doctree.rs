//! This module is used to store stuff from Rust's AST in a more convenient
//! manner (and with prettier names) before cleaning.
crate use self::StructType::*;

use rustc_span::{self, Span, Symbol};

use rustc_hir as hir;

crate struct Module<'hir> {
    crate name: Option<Symbol>,
    crate where_outer: Span,
    crate where_inner: Span,
    crate mods: Vec<Module<'hir>>,
    crate id: hir::HirId,
    // (item, renamed)
    crate items: Vec<(&'hir hir::Item<'hir>, Option<Symbol>)>,
    crate foreigns: Vec<(&'hir hir::ForeignItem<'hir>, Option<Symbol>)>,
    crate macros: Vec<(&'hir hir::MacroDef<'hir>, Option<Symbol>)>,
    crate is_crate: bool,
}

impl Module<'hir> {
    crate fn new(name: Option<Symbol>) -> Module<'hir> {
        Module {
            name,
            id: hir::CRATE_HIR_ID,
            where_outer: rustc_span::DUMMY_SP,
            where_inner: rustc_span::DUMMY_SP,
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

crate fn struct_type_from_def(vdata: &hir::VariantData<'_>) -> StructType {
    match *vdata {
        hir::VariantData::Struct(..) => Plain,
        hir::VariantData::Tuple(..) => Tuple,
        hir::VariantData::Unit(..) => Unit,
    }
}
