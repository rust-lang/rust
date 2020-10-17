use rustc_hir::def_id::{DefId, DefIdSet};
use rustc_middle::middle::privacy::AccessLevels;
use std::mem;

use crate::clean::{self, GetDefId, Item};
use crate::fold::{DocFolder, StripItem};

pub struct Stripper<'a> {
    pub retained: &'a mut DefIdSet,
    pub access_levels: &'a AccessLevels<DefId>,
    pub update_retained: bool,
}

impl<'a> DocFolder for Stripper<'a> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        match i.inner {
            clean::StrippedItem(..) => {
                // We need to recurse into stripped modules to strip things
                // like impl methods but when doing so we must not add any
                // items to the `retained` set.
                debug!("Stripper: recursing into stripped {:?} {:?}", i.type_(), i.name);
                let old = mem::replace(&mut self.update_retained, false);
                let ret = self.fold_item_recur(i);
                self.update_retained = old;
                return ret;
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
            | clean::ForeignTypeItem => {
                if i.def_id.is_local() {
                    if !self.access_levels.is_exported(i.def_id) {
                        debug!("Stripper: stripping {:?} {:?}", i.type_(), i.name);
                        return None;
                    }
                }
            }

            clean::StructFieldItem(..) => {
                if i.visibility != clean::Public {
                    return StripItem(i).strip();
                }
            }

            clean::ModuleItem(..) => {
                if i.def_id.is_local() && i.visibility != clean::Public {
                    debug!("Stripper: stripping module {:?}", i.name);
                    let old = mem::replace(&mut self.update_retained, false);
                    let ret = StripItem(self.fold_item_recur(i).unwrap()).strip();
                    self.update_retained = old;
                    return ret;
                }
            }

            // handled in the `strip-priv-imports` pass
            clean::ExternCrateItem(..) | clean::ImportItem(..) => {}

            clean::ImplItem(..) => {}

            // tymethods/macros have no control over privacy
            clean::MacroItem(..) | clean::TyMethodItem(..) => {}

            // Proc-macros are always public
            clean::ProcMacroItem(..) => {}

            // Primitives are never stripped
            clean::PrimitiveItem(..) => {}

            // Associated types are never stripped
            clean::AssocTypeItem(..) => {}

            // Keywords are never stripped
            clean::KeywordItem(..) => {}
        }

        let fastreturn = match i.inner {
            // nothing left to do for traits (don't want to filter their
            // methods out, visibility controlled by the trait)
            clean::TraitItem(..) => true,

            // implementations of traits are always public.
            clean::ImplItem(ref imp) if imp.trait_.is_some() => true,
            // Struct variant fields have inherited visibility
            clean::VariantItem(clean::Variant { kind: clean::VariantKind::Struct(..) }) => true,
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

        if let Some(ref i) = i {
            if self.update_retained {
                self.retained.insert(i.def_id);
            }
        }
        i
    }
}

/// This stripper discards all impls which reference stripped items
pub struct ImplStripper<'a> {
    pub retained: &'a DefIdSet,
}

impl<'a> DocFolder for ImplStripper<'a> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        if let clean::ImplItem(ref imp) = i.inner {
            // emptied none trait impls can be stripped
            if imp.trait_.is_none() && imp.items.is_empty() {
                return None;
            }
            if let Some(did) = imp.for_.def_id() {
                if did.is_local() && !imp.for_.is_generic() && !self.retained.contains(&did) {
                    debug!("ImplStripper: impl item for stripped type; removing");
                    return None;
                }
            }
            if let Some(did) = imp.trait_.def_id() {
                if did.is_local() && !self.retained.contains(&did) {
                    debug!("ImplStripper: impl item for stripped trait; removing");
                    return None;
                }
            }
            if let Some(generics) = imp.trait_.as_ref().and_then(|t| t.generics()) {
                for typaram in generics {
                    if let Some(did) = typaram.def_id() {
                        if did.is_local() && !self.retained.contains(&did) {
                            debug!(
                                "ImplStripper: stripped item in trait's generics; removing impl"
                            );
                            return None;
                        }
                    }
                }
            }
        }
        self.fold_item_recur(i)
    }
}

/// This stripper discards all private import statements (`use`, `extern crate`)
pub struct ImportStripper;

impl DocFolder for ImportStripper {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        match i.inner {
            clean::ExternCrateItem(..) | clean::ImportItem(..) if i.visibility != clean::Public => {
                None
            }
            _ => self.fold_item_recur(i),
        }
    }
}
