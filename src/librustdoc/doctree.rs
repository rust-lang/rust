//! This module is used to store stuff from Rust's AST in a more convenient
//! manner (and with prettier names) before cleaning.
use crate::core::DocContext;
use rustc_hir as hir;
use rustc_span::{self, Span, Symbol};

/// A wrapper around a [`hir::Item`].
#[derive(Debug)]
crate struct Item<'hir> {
    /// the wrapped item
    crate hir_item: &'hir hir::Item<'hir>,
    /// the explicit renamed name
    crate renamed_name: Option<Symbol>,
    /// whether the item is from a glob import
    /// if `from_glob` is true and we see another item with same name and namespace,
    /// then this item can be replaced with that one
    crate from_glob: bool,
}

impl<'hir> Item<'hir> {
    pub(crate) fn new(
        hir_item: &'hir hir::Item<'hir>,
        renamed_name: Option<Symbol>,
        from_glob: bool,
    ) -> Self {
        Self { hir_item, renamed_name, from_glob }
    }

    pub(crate) fn name(&self) -> Symbol {
        self.renamed_name.unwrap_or(self.hir_item.ident.name)
    }
}

crate struct Module<'hir> {
    crate name: Symbol,
    crate where_outer: Span,
    crate where_inner: Span,
    crate mods: Vec<Module<'hir>>,
    crate id: hir::HirId,
    crate items: Vec<Item<'hir>>,
    crate foreigns: Vec<(&'hir hir::ForeignItem<'hir>, Option<Symbol>)>,
    crate macros: Vec<(&'hir hir::MacroDef<'hir>, Option<Symbol>)>,
    crate is_crate: bool,
    /// whether the module is from a glob import
    /// if `from_glob` is true and we see another module with same name,
    /// then this module can be replaced with that one
    crate from_glob: bool,
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

    pub(crate) fn push_item(&mut self, ctx: &DocContext<'_>, new_item: Item<'hir>) {
        let new_item_ns = ctx.tcx.def_kind(new_item.hir_item.def_id).ns();
        let mut to_remove = None;
        if !new_item.name().is_empty() && new_item_ns.is_some() {
            if let Some((index, existing_item)) =
                self.items.iter_mut().enumerate().find(|(_, item)| {
                    item.name() == new_item.name()
                        && ctx.tcx.def_kind(item.hir_item.def_id).ns() == new_item_ns
                })
            {
                match (existing_item.from_glob, new_item.from_glob) {
                    (true, false) => {
                        // `existing_item` is from glob, `new_item` is not,
                        // `new_item` should shadow `existing_item`
                        debug!("push_item: {:?} shadowed by {:?}", existing_item, new_item);
                        *existing_item = new_item;
                        return;
                    }
                    (false, true) => {
                        // `existing_item` is not from glob but `new_item` is,
                        // just keep `existing_item` and return at once
                        return;
                    }
                    (true, true) => {
                        // both `existing_item` and `new_item` are from glob
                        if existing_item.hir_item.def_id == new_item.hir_item.def_id {
                            // they actually point to same object, it is ok to just keep one of them
                            return;
                        } else {
                            // a "glob import vs glob import in the same module" will raise when actually use these item
                            // we should accept neither of these two items
                            to_remove = Some(index);
                        }
                    }
                    (false, false) => {
                        // should report "defined multiple time" error before reach this
                        unreachable!()
                    }
                }
            }
        }
        if let Some(index) = to_remove {
            self.items.swap_remove(index);
            return;
        }
        // no item with same name and namespace exists, just collect `new_item`
        self.items.push(new_item);
    }

    pub(crate) fn push_mod(&mut self, new_mod: Module<'hir>) {
        let mut to_remove = None;
        if let Some((index, existing_mod)) =
            self.mods.iter_mut().enumerate().find(|(_, mod_)| mod_.name == new_mod.name)
        {
            match (existing_mod.from_glob, new_mod.from_glob) {
                (true, false) => {
                    // `existing_mod` is from glob, no matter whether `new_mod` is from glob,
                    // `new_mod` should always shadow `existing_mod`
                    debug!("push_mod: {:?} shadowed by {:?}", existing_mod.name, new_mod.name);
                    *existing_mod = new_mod;
                    return;
                }
                (true, true) => {
                    // both `existing_item` and `new_item` are from glob
                    if existing_mod.id == new_mod.id {
                        // they actually point to same object, it is ok to just keep one of them
                        return;
                    } else {
                        // a "glob import vs glob import in the same module" will raise when actually use these item
                        // we should accept neither of these two items
                        to_remove = Some(index);
                    }
                }
                (false, true) => {
                    // `existing_mod` is not from glob but `new_mod` is,
                    // just keep `existing_mod` and return at once
                    return;
                }
                (false, false) => {
                    // should report "defined multiple time" error before reach this
                    unreachable!()
                }
            }
        }
        if let Some(index) = to_remove {
            self.mods.swap_remove(index);
            return;
        }
        // no mod with same name exists, just collect `new_mod`
        self.mods.push(new_mod);
    }
}
