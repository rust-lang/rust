//! A collection of utility functions for the `strip_*` passes.

use std::mem;

use rustc_hir::def_id::DefId;
use rustc_middle::ty::{TyCtxt, Visibility};
use tracing::debug;

use crate::clean::utils::inherits_doc_hidden;
use crate::clean::{self, Item, ItemId, ItemIdSet};
use crate::fold::{DocFolder, strip_item};
use crate::formats::cache::Cache;
use crate::visit_lib::RustdocEffectiveVisibilities;

pub(crate) struct Stripper<'a, 'tcx> {
    pub(crate) retained: &'a mut ItemIdSet,
    pub(crate) effective_visibilities: &'a RustdocEffectiveVisibilities,
    pub(crate) update_retained: bool,
    pub(crate) is_json_output: bool,
    pub(crate) tcx: TyCtxt<'tcx>,
}

// We need to handle this differently for the JSON output because some non exported items could
// be used in public API. And so, we need these items as well. `is_exported` only checks if they
// are in the public API, which is not enough.
#[inline]
fn is_item_reachable(
    tcx: TyCtxt<'_>,
    is_json_output: bool,
    effective_visibilities: &RustdocEffectiveVisibilities,
    item_id: ItemId,
) -> bool {
    if is_json_output {
        effective_visibilities.is_reachable(tcx, item_id.expect_def_id())
    } else {
        effective_visibilities.is_exported(tcx, item_id.expect_def_id())
    }
}

impl DocFolder for Stripper<'_, '_> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        match i.kind {
            clean::StrippedItem(..) => {
                // We need to recurse into stripped modules to strip things
                // like impl methods but when doing so we must not add any
                // items to the `retained` set.
                debug!("Stripper: recursing into stripped {:?} {:?}", i.type_(), i.name);
                let old = mem::replace(&mut self.update_retained, false);
                let ret = self.fold_item_recur(i);
                self.update_retained = old;
                return Some(ret);
            }
            // These items can all get re-exported
            clean::TypeAliasItem(..)
            | clean::StaticItem(..)
            | clean::StructItem(..)
            | clean::EnumItem(..)
            | clean::TraitItem(..)
            | clean::FunctionItem(..)
            | clean::VariantItem(..)
            | clean::ForeignFunctionItem(..)
            | clean::ForeignStaticItem(..)
            | clean::ConstantItem(..)
            | clean::UnionItem(..)
            | clean::TraitAliasItem(..)
            | clean::MacroItem(..)
            | clean::ForeignTypeItem => {
                let item_id = i.item_id;
                if item_id.is_local()
                    && !is_item_reachable(
                        self.tcx,
                        self.is_json_output,
                        self.effective_visibilities,
                        item_id,
                    )
                {
                    debug!("Stripper: stripping {:?} {:?}", i.type_(), i.name);
                    return None;
                }
            }

            clean::MethodItem(..)
            | clean::ProvidedAssocConstItem(..)
            | clean::ImplAssocConstItem(..)
            | clean::AssocTypeItem(..) => {
                let item_id = i.item_id;
                if item_id.is_local()
                    && !self.effective_visibilities.is_reachable(self.tcx, item_id.expect_def_id())
                {
                    debug!("Stripper: stripping {:?} {:?}", i.type_(), i.name);
                    return None;
                }
            }

            clean::StructFieldItem(..) => {
                if i.visibility(self.tcx) != Some(Visibility::Public) {
                    return Some(strip_item(i));
                }
            }

            clean::ModuleItem(..) => {
                if i.item_id.is_local()
                    && !is_item_reachable(
                        self.tcx,
                        self.is_json_output,
                        self.effective_visibilities,
                        i.item_id,
                    )
                {
                    debug!("Stripper: stripping module {:?}", i.name);
                    let old = mem::replace(&mut self.update_retained, false);
                    let ret = strip_item(self.fold_item_recur(i));
                    self.update_retained = old;
                    return Some(ret);
                }
            }

            // handled in the `strip-priv-imports` pass
            clean::ExternCrateItem { .. } | clean::ImportItem(_) => {}

            clean::ImplItem(..) => {}

            // tymethods etc. have no control over privacy
            clean::RequiredMethodItem(..)
            | clean::RequiredAssocConstItem(..)
            | clean::RequiredAssocTypeItem(..) => {}

            // Proc-macros are always public
            clean::ProcMacroItem(..) => {}

            // Primitives are never stripped
            clean::PrimitiveItem(..) => {}

            // Keywords are never stripped
            clean::KeywordItem => {}
            // Attributes are never stripped
            clean::AttributeItem => {}
        }

        let fastreturn = match i.kind {
            // nothing left to do for traits (don't want to filter their
            // methods out, visibility controlled by the trait)
            clean::TraitItem(..) => true,

            // implementations of traits are always public.
            clean::ImplItem(ref imp) if imp.trait_.is_some() => true,
            // Variant fields have inherited visibility
            clean::VariantItem(clean::Variant {
                kind: clean::VariantKind::Struct(..) | clean::VariantKind::Tuple(..),
                ..
            }) => true,
            _ => false,
        };

        let i = if fastreturn {
            if self.update_retained {
                self.retained.insert(i.item_id);
            }
            return Some(i);
        } else {
            self.fold_item_recur(i)
        };

        if self.update_retained {
            self.retained.insert(i.item_id);
        }
        Some(i)
    }
}

/// This stripper discards all impls which reference stripped items
pub(crate) struct ImplStripper<'a, 'tcx> {
    pub(crate) tcx: TyCtxt<'tcx>,
    pub(crate) retained: &'a ItemIdSet,
    pub(crate) cache: &'a Cache,
    pub(crate) is_json_output: bool,
    pub(crate) document_private: bool,
    pub(crate) document_hidden: bool,
}

impl ImplStripper<'_, '_> {
    #[inline]
    fn should_keep_impl(&self, item: &Item, for_def_id: DefId) -> bool {
        if !for_def_id.is_local() || self.retained.contains(&for_def_id.into()) {
            true
        } else if self.is_json_output {
            // If the "for" item is exported and the impl block isn't `#[doc(hidden)]`, then we
            // need to keep it.
            self.cache.effective_visibilities.is_exported(self.tcx, for_def_id)
                && (self.document_hidden
                    || ((!item.is_doc_hidden()
                        && for_def_id
                            .as_local()
                            .map(|def_id| !inherits_doc_hidden(self.tcx, def_id, None))
                            .unwrap_or(true))
                        || self.cache.inlined_items.contains(&for_def_id)))
        } else {
            false
        }
    }
}

impl DocFolder for ImplStripper<'_, '_> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        if let clean::ImplItem(ref imp) = i.kind {
            // Impl blocks can be skipped if they are: empty; not a trait impl; and have no
            // documentation.
            //
            // There is one special case: if the impl block contains only private items.
            if imp.trait_.is_none() {
                // If the only items present are private ones and we're not rendering private items,
                // we don't document it.
                if !imp.items.is_empty()
                    && !self.document_private
                    && imp.items.iter().all(|i| {
                        let item_id = i.item_id;
                        item_id.is_local()
                            && !self
                                .cache
                                .effective_visibilities
                                .is_reachable(self.tcx, item_id.expect_def_id())
                    })
                {
                    debug!("ImplStripper: no public item; removing {imp:?}");
                    return None;
                } else if imp.items.is_empty() && i.doc_value().is_empty() {
                    debug!("ImplStripper: no item and no doc; removing {imp:?}");
                    return None;
                }
            }
            // Because we don't inline in `maybe_inline_local` if the output format is JSON,
            // we need to make a special check for JSON output: we want to keep it unless it has
            // a `#[doc(hidden)]` attribute if the `for_` type is exported.
            if let Some(did) = imp.for_.def_id(self.cache)
                && !imp.for_.is_assoc_ty()
                && !self.should_keep_impl(&i, did)
            {
                debug!("ImplStripper: impl item for stripped type; removing {imp:?}");
                return None;
            }
            if let Some(did) = imp.trait_.as_ref().map(|t| t.def_id())
                && !self.should_keep_impl(&i, did)
            {
                debug!("ImplStripper: impl item for stripped trait; removing {imp:?}");
                return None;
            }
            if let Some(generics) = imp.trait_.as_ref().and_then(|t| t.generics()) {
                for typaram in generics {
                    if let Some(did) = typaram.def_id(self.cache)
                        && !self.should_keep_impl(&i, did)
                    {
                        debug!("ImplStripper: stripped item in trait's generics; removing {imp:?}");
                        return None;
                    }
                }
            }
        }
        Some(self.fold_item_recur(i))
    }
}

/// This stripper discards all private import statements (`use`, `extern crate`)
pub(crate) struct ImportStripper<'tcx> {
    pub(crate) tcx: TyCtxt<'tcx>,
    pub(crate) is_json_output: bool,
    pub(crate) document_hidden: bool,
}

impl ImportStripper<'_> {
    fn import_should_be_hidden(&self, i: &Item, imp: &clean::Import) -> bool {
        if self.is_json_output {
            // FIXME: This should be handled the same way as for HTML output.
            imp.imported_item_is_doc_hidden(self.tcx)
        } else {
            i.is_doc_hidden()
        }
    }
}

impl DocFolder for ImportStripper<'_> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        match &i.kind {
            clean::ImportItem(imp)
                if !self.document_hidden && self.import_should_be_hidden(&i, imp) =>
            {
                None
            }
            // clean::ImportItem(_) if !self.document_hidden && i.is_doc_hidden() => None,
            clean::ExternCrateItem { .. } | clean::ImportItem(..)
                if i.visibility(self.tcx) != Some(Visibility::Public) =>
            {
                None
            }
            _ => Some(self.fold_item_recur(i)),
        }
    }
}
