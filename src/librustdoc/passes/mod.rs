// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::def_id::DefId;
use rustc::middle::privacy::AccessLevels;
use rustc::util::nodemap::DefIdSet;
use std::mem;

use clean::{self, GetDefId, Item};
use fold;
use fold::FoldItem::Strip;
use plugins;

mod collapse_docs;
pub use self::collapse_docs::collapse_docs;

mod strip_hidden;
pub use self::strip_hidden::strip_hidden;

mod strip_private;
pub use self::strip_private::strip_private;

mod strip_priv_imports;
pub use self::strip_priv_imports::strip_priv_imports;

mod unindent_comments;
pub use self::unindent_comments::unindent_comments;

type Pass = (&'static str,                                      // name
             fn(clean::Crate) -> plugins::PluginResult,         // fn
             &'static str);                                     // description

pub const PASSES: &'static [Pass] = &[
    ("strip-hidden", strip_hidden,
     "strips all doc(hidden) items from the output"),
    ("unindent-comments", unindent_comments,
     "removes excess indentation on comments in order for markdown to like it"),
    ("collapse-docs", collapse_docs,
     "concatenates all document attributes into one document attribute"),
    ("strip-private", strip_private,
     "strips all private items from a crate which cannot be seen externally, \
      implies strip-priv-imports"),
    ("strip-priv-imports", strip_priv_imports,
     "strips all private import statements (`use`, `extern crate`) from a crate"),
];

pub const DEFAULT_PASSES: &'static [&'static str] = &[
    "strip-hidden",
    "strip-private",
    "collapse-docs",
    "unindent-comments",
];


struct Stripper<'a> {
    retained: &'a mut DefIdSet,
    access_levels: &'a AccessLevels<DefId>,
    update_retained: bool,
}

impl<'a> fold::DocFolder for Stripper<'a> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        match i.inner {
            clean::StrippedItem(..) => {
                // We need to recurse into stripped modules to strip things
                // like impl methods but when doing so we must not add any
                // items to the `retained` set.
                let old = mem::replace(&mut self.update_retained, false);
                let ret = self.fold_item_recur(i);
                self.update_retained = old;
                return ret;
            }
            // These items can all get re-exported
            clean::TypedefItem(..) | clean::StaticItem(..) |
            clean::StructItem(..) | clean::EnumItem(..) |
            clean::TraitItem(..) | clean::FunctionItem(..) |
            clean::VariantItem(..) | clean::MethodItem(..) |
            clean::ForeignFunctionItem(..) | clean::ForeignStaticItem(..) |
            clean::ConstantItem(..) | clean::UnionItem(..) => {
                if i.def_id.is_local() {
                    if !self.access_levels.is_exported(i.def_id) {
                        return None;
                    }
                }
            }

            clean::StructFieldItem(..) => {
                if i.visibility != Some(clean::Public) {
                    return Strip(i).fold();
                }
            }

            clean::ModuleItem(..) => {
                if i.def_id.is_local() && i.visibility != Some(clean::Public) {
                    let old = mem::replace(&mut self.update_retained, false);
                    let ret = Strip(self.fold_item_recur(i).unwrap()).fold();
                    self.update_retained = old;
                    return ret;
                }
            }

            // handled in the `strip-priv-imports` pass
            clean::ExternCrateItem(..) | clean::ImportItem(..) => {}

            clean::DefaultImplItem(..) | clean::ImplItem(..) => {}

            // tymethods/macros have no control over privacy
            clean::MacroItem(..) | clean::TyMethodItem(..) => {}

            // Primitives are never stripped
            clean::PrimitiveItem(..) => {}

            // Associated consts and types are never stripped
            clean::AssociatedConstItem(..) |
            clean::AssociatedTypeItem(..) => {}
        }

        let fastreturn = match i.inner {
            // nothing left to do for traits (don't want to filter their
            // methods out, visibility controlled by the trait)
            clean::TraitItem(..) => true,

            // implementations of traits are always public.
            clean::ImplItem(ref imp) if imp.trait_.is_some() => true,
            // Struct variant fields have inherited visibility
            clean::VariantItem(clean::Variant {
                kind: clean::VariantKind::Struct(..)
            }) => true,
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

        i.and_then(|i| {
            match i.inner {
                // emptied modules have no need to exist
                clean::ModuleItem(ref m)
                    if m.items.is_empty() &&
                       i.doc_value().is_none() => None,
                _ => {
                    if self.update_retained {
                        self.retained.insert(i.def_id);
                    }
                    Some(i)
                }
            }
        })
    }
}

// This stripper discards all impls which reference stripped items
struct ImplStripper<'a> {
    retained: &'a DefIdSet
}

impl<'a> fold::DocFolder for ImplStripper<'a> {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        if let clean::ImplItem(ref imp) = i.inner {
            // emptied none trait impls can be stripped
            if imp.trait_.is_none() && imp.items.is_empty() {
                return None;
            }
            if let Some(did) = imp.for_.def_id() {
                if did.is_local() && !imp.for_.is_generic() &&
                    !self.retained.contains(&did)
                {
                    return None;
                }
            }
            if let Some(did) = imp.trait_.def_id() {
                if did.is_local() && !self.retained.contains(&did) {
                    return None;
                }
            }
        }
        self.fold_item_recur(i)
    }
}

// This stripper discards all private import statements (`use`, `extern crate`)
struct ImportStripper;
impl fold::DocFolder for ImportStripper {
    fn fold_item(&mut self, i: Item) -> Option<Item> {
        match i.inner {
            clean::ExternCrateItem(..) |
            clean::ImportItem(..) if i.visibility != Some(clean::Public) => None,
            _ => self.fold_item_recur(i)
        }
    }
}
