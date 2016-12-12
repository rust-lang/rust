// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::middle::cstore::CrateStore;
use rustc::middle::privacy::{AccessLevels, AccessLevel};
use rustc::hir::def::Def;
use rustc::hir::def_id::{CrateNum, CRATE_DEF_INDEX, DefId};
use rustc::ty::Visibility;

use std::cell::RefMut;

use clean::{AttributesExt, NestedAttributesExt};

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

    pub fn visit_lib(&mut self, cnum: CrateNum) {
        let did = DefId { krate: cnum, index: CRATE_DEF_INDEX };
        self.update(did, Some(AccessLevel::Public));
        self.visit_mod(did);
    }

    // Updates node level and returns the updated level
    fn update(&mut self, did: DefId, level: Option<AccessLevel>) -> Option<AccessLevel> {
        let is_hidden = self.cx.tcx.get_attrs(did).lists("doc").has_word("hidden");

        let old_level = self.access_levels.map.get(&did).cloned();
        // Accessibility levels can only grow
        if level > old_level && !is_hidden {
            self.access_levels.map.insert(did, level.unwrap());
            level
        } else {
            old_level
        }
    }

    pub fn visit_mod(&mut self, def_id: DefId) {
        for item in self.cstore.item_children(def_id) {
            self.visit_item(item.def);
        }
    }

    fn visit_item(&mut self, def: Def) {
        let def_id = def.def_id();
        let vis = self.cstore.visibility(def_id);
        let inherited_item_level = if vis == Visibility::Public {
            self.prev_level
        } else {
            None
        };

        let item_level = self.update(def_id, inherited_item_level);

        if let Def::Mod(..) = def {
            let orig_level = self.prev_level;

            self.prev_level = item_level;
            self.visit_mod(def_id);
            self.prev_level = orig_level;
        }
    }
}
