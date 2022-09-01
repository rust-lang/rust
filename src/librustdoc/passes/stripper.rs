//! A collection of utility functions for the `strip_*` passes.
use rustc_hir::def_id::DefId;
use rustc_middle::middle::privacy::AccessLevels;
use std::mem;

use crate::clean::{self, Item, ItemId, ItemIdSet};
use crate::fold::{strip_item, DocFolder};
use crate::formats::cache::Cache;

pub(crate) struct Stripper<'a> {
    pub(crate) retained: &'a mut ItemIdSet,
    pub(crate) access_levels: &'a AccessLevels<DefId>,
    pub(crate) update_retained: bool,
    pub(crate) is_json_output: bool,
}

// We need to handle this differently for the JSON output because some non exported items could
// be used in public API. And so, we need these items as well. `is_exported` only checks if they
// are in the public API, which is not enough.
#[inline]
fn is_item_reachable(
    is_json_output: bool,
    access_levels: &AccessLevels<DefId>,
    item_id: ItemId,
) -> bool {
    if is_json_output {
        access_levels.is_reachable(item_id.expect_def_id())
    } else {
        access_levels.is_exported(item_id.expect_def_id())
    }
}

impl<'a> DocFolder for Stripper<'a> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        match *i.kind {
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
            clean::OpaqueTyItem(..)
            | clean::TypedefItem(..)
            | clean::StaticItem(..)
            | clean::StructItem(..)
            | clean::EnumItem(..)
            | clean::TraitItem(..)
            | clean::FunctionItem(..)
            | clean::VariantItem(..)
            | clean::MethodItem(..)
            | clean::ForeignFunctionItem(..)
            | clean::ForeignStaticItem(..)
            | clean::ConstantItem(..)
            | clean::UnionItem(..)
            | clean::AssocConstItem(..)
            | clean::AssocTypeItem(..)
            | clean::TraitAliasItem(..)
            | clean::MacroItem(..)
            | clean::ForeignTypeItem => {
                let item_id = i.item_id;
                if item_id.is_local()
                    && !is_item_reachable(self.is_json_output, self.access_levels, item_id)
                {
                    debug!("Stripper: stripping {:?} {:?}", i.type_(), i.name);
                    return None;
                }
            }

            clean::StructFieldItem(..) => {
                if !i.visibility.is_public() {
                    return Some(strip_item(i));
                }
            }

            clean::ModuleItem(..) => {
                if i.item_id.is_local() && !i.visibility.is_public() {
                    debug!("Stripper: stripping module {:?}", i.name);
                    let old = mem::replace(&mut self.update_retained, false);
                    let ret = strip_item(self.fold_item_recur(i));
                    self.update_retained = old;
                    return Some(ret);
                }
            }

            // handled in the `strip-priv-imports` pass
            clean::ExternCrateItem { .. } => {}
            clean::ImportItem(ref imp) => {
                // Because json doesn't inline imports from private modules, we need to mark
                // the imported item as retained so it's impls won't be stripped.
                //
                // FIXME: Is it necessary to check for json output here: See
                // https://github.com/rust-lang/rust/pull/100325#discussion_r941495215
                if let Some(did) = imp.source.did && self.is_json_output {
                    self.retained.insert(did.into());
                }
            }

            clean::ImplItem(..) => {}

            // tymethods etc. have no control over privacy
            clean::TyMethodItem(..) | clean::TyAssocConstItem(..) | clean::TyAssocTypeItem(..) => {}

            // Proc-macros are always public
            clean::ProcMacroItem(..) => {}

            // Primitives are never stripped
            clean::PrimitiveItem(..) => {}

            // Keywords are never stripped
            clean::KeywordItem => {}
        }

        let fastreturn = match *i.kind {
            // nothing left to do for traits (don't want to filter their
            // methods out, visibility controlled by the trait)
            clean::TraitItem(..) => true,

            // implementations of traits are always public.
            clean::ImplItem(ref imp) if imp.trait_.is_some() => true,
            // Variant fields have inherited visibility
            clean::VariantItem(clean::Variant::Struct(..) | clean::Variant::Tuple(..)) => true,
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
pub(crate) struct ImplStripper<'a> {
    pub(crate) retained: &'a ItemIdSet,
    pub(crate) cache: &'a Cache,
    pub(crate) is_json_output: bool,
    pub(crate) document_private: bool,
}

impl<'a> DocFolder for ImplStripper<'a> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        if let clean::ImplItem(ref imp) = *i.kind {
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
                            && !is_item_reachable(
                                self.is_json_output,
                                &self.cache.access_levels,
                                item_id,
                            )
                    })
                {
                    return None;
                } else if imp.items.is_empty() && i.doc_value().is_none() {
                    return None;
                }
            }
            if let Some(did) = imp.for_.def_id(self.cache) {
                if did.is_local() && !imp.for_.is_assoc_ty() && !self.retained.contains(&did.into())
                {
                    debug!("ImplStripper: impl item for stripped type; removing");
                    return None;
                }
            }
            if let Some(did) = imp.trait_.as_ref().map(|t| t.def_id()) {
                if did.is_local() && !self.retained.contains(&did.into()) {
                    debug!("ImplStripper: impl item for stripped trait; removing");
                    return None;
                }
            }
            if let Some(generics) = imp.trait_.as_ref().and_then(|t| t.generics()) {
                for typaram in generics {
                    if let Some(did) = typaram.def_id(self.cache) {
                        if did.is_local() && !self.retained.contains(&did.into()) {
                            debug!(
                                "ImplStripper: stripped item in trait's generics; removing impl"
                            );
                            return None;
                        }
                    }
                }
            }
        }
        Some(self.fold_item_recur(i))
    }
}

/// This stripper discards all private import statements (`use`, `extern crate`)
pub(crate) struct ImportStripper;

impl DocFolder for ImportStripper {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        match *i.kind {
            clean::ExternCrateItem { .. } | clean::ImportItem(..) if !i.visibility.is_public() => {
                None
            }
            _ => Some(self.fold_item_recur(i)),
        }
    }
}
