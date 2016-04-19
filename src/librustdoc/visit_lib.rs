// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::middle::cstore::{CrateStore, ChildItem, DefLike};
use rustc::middle::privacy::{AccessLevels, AccessLevel};
use rustc::hir::def::Def;
use rustc::hir::def_id::{CRATE_DEF_INDEX, DefId};
use rustc::ty::Visibility;
use syntax::ast;

use std::cell::RefMut;

use clean::{Attributes, Clean};

// FIXME: this may not be exhaustive, but is sufficient for rustdocs current uses

/// Similar to `librustc_privacy::EmbargoVisitor`, but also takes
/// specific rustdoc annotations into account (i.e. `doc(hidden)`)
pub struct LibEmbargoVisitor<'a, 'b: 'a, 'tcx: 'b> {
    cx: &'a ::core::DocContext<'b, 'tcx>,
    cstore: &'a CrateStore<'tcx>,
    // Accessibility levels for reachable nodes
    access_levels: RefMut<'a, AccessLevels<DefId>>,
    // Previous accessibility level, None means unreachable
    prev_level: Option<AccessLevel>,
}

impl<'a, 'b, 'tcx> LibEmbargoVisitor<'a, 'b, 'tcx> {
    pub fn new(cx: &'a ::core::DocContext<'b, 'tcx>) -> LibEmbargoVisitor<'a, 'b, 'tcx> {
        LibEmbargoVisitor {
            cx: cx,
            cstore: &*cx.sess().cstore,
            access_levels: cx.access_levels.borrow_mut(),
            prev_level: Some(AccessLevel::Public),
        }
    }

    pub fn visit_lib(&mut self, cnum: ast::CrateNum) {
        let did = DefId { krate: cnum, index: CRATE_DEF_INDEX };
        self.visit_mod(did);
    }

    // Updates node level and returns the updated level
    fn update(&mut self, did: DefId, level: Option<AccessLevel>) -> Option<AccessLevel> {
        let attrs: Vec<_> = self.cx.tcx().get_attrs(did).iter()
                                                        .map(|a| a.clean(self.cx))
                                                        .collect();
        let is_hidden = attrs.list("doc").has_word("hidden");

        let old_level = self.access_levels.map.get(&did).cloned();
        // Accessibility levels can only grow
        if level > old_level && !is_hidden {
            self.access_levels.map.insert(did, level.unwrap());
            level
        } else {
            old_level
        }
    }

    pub fn visit_mod(&mut self, did: DefId) {
        for item in self.cstore.item_children(did) {
            if let DefLike::DlDef(def) = item.def {
                match def {
                    Def::Mod(did) |
                    Def::ForeignMod(did) |
                    Def::Trait(did) |
                    Def::Struct(did) |
                    Def::Enum(did) |
                    Def::TyAlias(did) |
                    Def::Fn(did) |
                    Def::Method(did) |
                    Def::Static(did, _) |
                    Def::Const(did) => self.visit_item(did, item),
                    _ => {}
                }
            }
        }
    }

    fn visit_item(&mut self, did: DefId, item: ChildItem) {
        let inherited_item_level = match item.def {
            DefLike::DlImpl(..) | DefLike::DlField => unreachable!(),
            DefLike::DlDef(def) => {
                match def {
                    Def::ForeignMod(..) => self.prev_level,
                    _ => if item.vis == Visibility::Public { self.prev_level } else { None }
                }
            }
        };

        let item_level = self.update(did, inherited_item_level);

        if let DefLike::DlDef(Def::Mod(did)) = item.def {
            let orig_level = self.prev_level;

            self.prev_level = item_level;
            self.visit_mod(did);
            self.prev_level = orig_level;
        }
    }
}
