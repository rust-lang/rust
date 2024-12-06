use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_span::{Symbol, sym};
use rustdoc_json_types::{self as types, Id}; // FIXME: Consistant.

use super::JsonRenderer;
use crate::clean::{self, ItemId};

pub(super) type IdInterner = FxHashMap<FullItemId, types::Id>;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
/// An uninterned id.
///
/// One of these coresponds to every:
/// 1. [`rustdoc_json_types::Item`].
/// 2. [`rustdoc_json_types::Id`] transitivly (as each `Item` has an `Id`).
///
/// It's *broadly* equivalent to a [`DefId`], but needs slightly more information
/// to fully disambiguate items, because sometimes we choose to split a single HIR
/// item into multiple JSON items, or have items with no coresponding HIR item.
pub(super) struct FullItemId {
    /// The "main" id of the item.
    ///
    /// In most cases this unequely identifies the item, other fields are just
    /// used for edge-cases.
    def_id: DefId,

    /// An extra DefId for auto-trait-impls or blanket-impls. These don't have DefId's
    /// as they're synthesized by rustdoc.
    extra_id: Option<DefId>,

    /// Needed for `rustc_doc_primitive` modules.
    ///
    /// For these, 1 DefId is used for both the primitive and the fake-module
    /// that holds it's docs.
    ///
    /// N.B. This only matters when documenting the standard library with
    /// `--document-private-items`. Maybe we should delete that module, and
    /// remove this.
    name: Option<Symbol>,

    /// Used to distinguish imports of different items with the same name.
    ///
    /// ```rust
    /// mod module {
    ///     pub struct Foo {}; // Exists in type namespace
    ///     pub fn Foo(){} // Exists in value namespace
    /// }
    ///
    /// pub use module::Foo; // Imports both items
    /// ```
    ///
    /// In HIR, the `pub use` is just 1 item, but in rustdoc-json it's 2, so
    /// we need to disambiguate.
    imported_id: Option<types::Id>,
}

impl JsonRenderer<'_> {
    pub(crate) fn id_from_item_default(&self, item_id: ItemId) -> Id {
        self.id_from_item_inner(item_id, None, None)
    }

    fn id_from_item_inner(
        &self,
        item_id: ItemId,
        name: Option<Symbol>,
        imported_id: Option<Id>,
    ) -> Id {
        let (def_id, extra_id) = match item_id {
            ItemId::DefId(did) => (did, None),
            ItemId::Blanket { for_, impl_id } => (for_, Some(impl_id)),
            ItemId::Auto { for_, trait_ } => (for_, Some(trait_)),
        };

        let name = match name {
            Some(name) => Some(name),
            None => {
                // We need this workaround because primitive types' DefId actually refers to
                // their parent module, which isn't present in the output JSON items. So
                // instead, we directly get the primitive symbol
                if matches!(self.tcx.def_kind(def_id), DefKind::Mod)
                    && let Some(prim) = self
                        .tcx
                        .get_attrs(def_id, sym::rustc_doc_primitive)
                        .find_map(|attr| attr.value_str())
                {
                    Some(prim)
                } else {
                    self.tcx.opt_item_name(def_id)
                }
            }
        };

        let key = FullItemId { def_id, extra_id, name, imported_id };

        let mut interner = self.id_interner.borrow_mut();
        let len = interner.len();
        *interner
            .entry(key)
            .or_insert_with(|| Id(len.try_into().expect("too many items in a crate")))
    }

    pub(crate) fn id_from_item(&self, item: &clean::Item) -> Id {
        match item.kind {
            clean::ItemKind::ImportItem(ref import) => {
                let extra =
                    import.source.did.map(ItemId::from).map(|i| self.id_from_item_default(i));
                self.id_from_item_inner(item.item_id, item.name, extra)
            }
            _ => self.id_from_item_inner(item.item_id, item.name, None),
        }
    }
}
