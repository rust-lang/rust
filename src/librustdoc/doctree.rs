//! This module is used to store stuff from Rust's AST in a more convenient
//! manner (and with prettier names) before cleaning.
use rustc_span::{self, Span, Symbol};

use rustc_hir as hir;

/// A warp around an hir::Item
#[derive(Debug)]
pub(crate) struct Item<'hir> {
    /// the wrapped item
    pub(crate) hir_item: &'hir hir::Item<'hir>,
    /// the explicit renamed name
    pub(crate) renamed_name: Option<Symbol>,
    /// whether the item is from a glob import
    /// if `from_glob` is true and we see another item with same name,
    /// then this item can be replaced with that one
    pub(crate) from_glob: bool,
}

impl<'hir> Item<'hir> {
    pub(crate) fn new(
        hir_item: &'hir hir::Item<'hir>,
        renamed_name: Option<Symbol>,
        from_glob: bool,
    ) -> Self {
        Self { hir_item, renamed_name, from_glob }
    }

    fn name(&'hir self) -> &'hir Symbol {
        self.renamed_name.as_ref().unwrap_or(&self.hir_item.ident.name)
    }
}

crate struct Module<'hir> {
    crate name: Symbol,
    crate where_outer: Span,
    crate where_inner: Span,
    crate mods: Vec<Module<'hir>>,
    crate id: hir::HirId,
    crate items: Vec<Item<'hir>>,
    // (item, renamed)
    crate foreigns: Vec<(&'hir hir::ForeignItem<'hir>, Option<Symbol>)>,
    crate macros: Vec<(&'hir hir::MacroDef<'hir>, Option<Symbol>)>,
    crate is_crate: bool,
    /// whether the module is from a glob import
    /// if `from_glob` is true and we see another module with same name,
    /// then this item can be replaced with that one
    pub(crate) from_glob: bool,
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
            from_glob: false,
        }
    }

    pub(crate) fn push_item(&mut self, new_item: Item<'hir>) {
        for item_iter in self.items.iter_mut() {
            if item_iter.name() == new_item.name() {
                if item_iter.from_glob {
                    debug!("push_item: {:?} shadowed by {:?}", *item_iter, new_item);
                    *item_iter = new_item;
                    return;
                } else if new_item.from_glob {
                    return;
                }
            }
        }
        self.items.push(new_item);
    }

    pub(crate) fn push_mod(&mut self, new_item: Module<'hir>) {
        for item_iter in self.mods.iter_mut() {
            if item_iter.name == new_item.name {
                if item_iter.from_glob {
                    debug!("push_mod: {:?} shadowed by {:?}", item_iter.name, new_item.name);
                    *item_iter = new_item;
                    return;
                } else if new_item.from_glob {
                    return;
                }
            }
        }
        self.mods.push(new_item);
    }
}
