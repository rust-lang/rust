use rustc::middle::privacy::{AccessLevels, AccessLevel};
use rustc::hir::def::Def;
use rustc::hir::def_id::{CrateNum, CRATE_DEF_INDEX, DefId};
use rustc::ty::Visibility;
use rustc::util::nodemap::FxHashSet;

use std::cell::RefMut;

use clean::{AttributesExt, NestedAttributesExt};

// FIXME: this may not be exhaustive, but is sufficient for rustdocs current uses

/// Similar to `librustc_privacy::EmbargoVisitor`, but also takes
/// specific rustdoc annotations into account (i.e., `doc(hidden)`)
pub struct LibEmbargoVisitor<'a, 'tcx: 'a, 'rcx: 'a> {
    cx: &'a ::core::DocContext<'a, 'tcx, 'rcx>,
    // Accessibility levels for reachable nodes
    access_levels: RefMut<'a, AccessLevels<DefId>>,
    // Previous accessibility level, None means unreachable
    prev_level: Option<AccessLevel>,
    // Keeps track of already visited modules, in case a module re-exports its parent
    visited_mods: FxHashSet<DefId>,
}

impl<'a, 'tcx, 'rcx> LibEmbargoVisitor<'a, 'tcx, 'rcx> {
    pub fn new(
        cx: &'a ::core::DocContext<'a, 'tcx, 'rcx>
    ) -> LibEmbargoVisitor<'a, 'tcx, 'rcx> {
        LibEmbargoVisitor {
            cx,
            access_levels: RefMut::map(cx.renderinfo.borrow_mut(), |ri| &mut ri.access_levels),
            prev_level: Some(AccessLevel::Public),
            visited_mods: FxHashSet::default()
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
        if !self.visited_mods.insert(def_id) {
            return;
        }

        for item in self.cx.tcx.item_children(def_id).iter() {
            if self.cx.tcx.def_key(item.def.def_id()).parent.map_or(false, |d| d == def_id.index) ||
                item.vis == Visibility::Public {
                self.visit_item(item.def);
            }
        }
    }

    fn visit_item(&mut self, def: Def) {
        let def_id = def.def_id();
        let vis = self.cx.tcx.visibility(def_id);
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
