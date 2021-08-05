//! This module is used to store stuff from Rust's AST in a more convenient
//! manner (and with prettier names) before cleaning.
use rustc_middle::ty::TyCtxt;
use rustc_span::{self, Span, Symbol};

use rustc_hir as hir;

#[derive(Debug)]
crate struct Module<'hir> {
    crate name: Symbol,
    crate where_inner: Span,
    crate mods: Vec<Module<'hir>>,
    crate id: hir::HirId,
    // (item, renamed)
    crate items: Vec<(&'hir hir::Item<'hir>, Option<Symbol>)>,
    crate foreigns: Vec<(&'hir hir::ForeignItem<'hir>, Option<Symbol>)>,
}

impl Module<'hir> {
    crate fn new(name: Symbol, id: hir::HirId, where_inner: Span) -> Module<'hir> {
        Module { name, id, where_inner, mods: Vec::new(), items: Vec::new(), foreigns: Vec::new() }
    }

    crate fn where_outer(&self, tcx: TyCtxt<'_>) -> Span {
        tcx.hir().span(self.id)
    }
}
