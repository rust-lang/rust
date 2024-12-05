use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_span::{Symbol, sym};
use rustdoc_json_types::{self as types, Id}; // FIXME: Consistant.

use super::JsonRenderer;
use crate::clean::{self, ItemId};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub(super) struct FullItemId {
    def_id: DefId,
    name: Option<Symbol>,
    /// Used to distinguish imports of different items with the same name
    extra: Option<types::Id>,
}

pub(super) type IdInterner = FxHashMap<(FullItemId, Option<FullItemId>), types::Id>;

impl JsonRenderer<'_> {
    pub(crate) fn id_from_item_default(&self, item_id: ItemId) -> Id {
        self.id_from_item_inner(item_id, None, None)
    }

    pub(crate) fn id_from_item_inner(
        &self,
        item_id: ItemId,
        name: Option<Symbol>,
        extra: Option<Id>,
    ) -> Id {
        let make_part = |def_id: DefId, name: Option<Symbol>, extra: Option<Id>| {
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

            FullItemId { def_id, name, extra }
        };

        let key = match item_id {
            ItemId::DefId(did) => (make_part(did, name, extra), None),
            ItemId::Blanket { for_, impl_id } => {
                (make_part(impl_id, None, None), Some(make_part(for_, name, extra)))
            }
            ItemId::Auto { for_, trait_ } => {
                (make_part(trait_, None, None), Some(make_part(for_, name, extra)))
            }
        };

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
