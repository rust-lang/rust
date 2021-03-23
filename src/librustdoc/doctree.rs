//! This module is used to store stuff from Rust's AST in a more convenient
//! manner (and with prettier names) before cleaning.
use rustc_span::{self, Span, Symbol};

use rustc_hir as hir;

crate struct Module<'hir> {
    crate name: Symbol,
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
    crate fn new(name: Symbol) -> Module<'hir> {
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
