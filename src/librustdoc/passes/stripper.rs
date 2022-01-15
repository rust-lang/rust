//! A collection of utility functions for the `strip_*` passes.
use rustc_hir::def_id::DefId;
use rustc_middle::middle::privacy::AccessLevels;
use std::mem;

use crate::clean::{self, Item, ItemIdSet};
use crate::fold::{strip_item, DocFolder};
use crate::formats::cache::Cache;

crate struct Stripper<'a> {
    crate retained: &'a mut ItemIdSet,
    crate access_levels: &'a AccessLevels<DefId>,
    crate update_retained: bool,
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
            | clean::TraitAliasItem(..)
            | clean::MacroItem(..)
            | clean::ForeignTypeItem => {
                if i.def_id.is_local() && !self.access_levels.is_exported(i.def_id.expect_def_id())
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
                if i.def_id.is_local() && !i.visibility.is_public() {
                    debug!("Stripper: stripping module {:?}", i.name);
                    let old = mem::replace(&mut self.update_retained, false);
                    let ret = strip_item(self.fold_item_recur(i));
                    self.update_retained = old;
                    return Some(ret);
                }
            }

            // handled in the `strip-priv-imports` pass
            clean::ExternCrateItem { .. } | clean::ImportItem(..) => {}

            clean::ImplItem(..) => {}

            // tymethods have no control over privacy
            clean::TyMethodItem(..) => {}

            // Proc-macros are always public
            clean::ProcMacroItem(..) => {}

            // Primitives are never stripped
            clean::PrimitiveItem(..) => {}

            // Associated types are never stripped
            clean::AssocTypeItem(..) => {}

            // Keywords are never stripped
            clean::KeywordItem(..) => {}
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
                self.retained.insert(i.def_id);
            }
            return Some(i);
        } else {
            self.fold_item_recur(i)
        };

        if self.update_retained {
            self.retained.insert(i.def_id);
        }
        Some(i)
    }
}

/// This stripper discards all impls which reference stripped items
crate struct ImplStripper<'a> {
    crate retained: &'a ItemIdSet,
    crate cache: &'a Cache,
}

impl<'a> DocFolder for ImplStripper<'a> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        if let clean::ImplItem(ref imp) = *i.kind {
            // emptied none trait impls can be stripped
            if imp.trait_.is_none() && imp.items.is_empty() {
                return None;
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
crate struct ImportStripper;

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
