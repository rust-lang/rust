use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, CRATE_DEF_INDEX};
use rustc_middle::middle::privacy::{AccessLevel, AccessLevels};
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::sym;

use crate::clean::{AttributesExt, NestedAttributesExt};

// FIXME: this may not be exhaustive, but is sufficient for rustdocs current uses

/// Similar to `librustc_privacy::EmbargoVisitor`, but also takes
/// specific rustdoc annotations into account (i.e., `doc(hidden)`)
crate struct LibEmbargoVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    // Accessibility levels for reachable nodes
    access_levels: &'a mut AccessLevels<DefId>,
    // Previous accessibility level, None means unreachable
    prev_level: Option<AccessLevel>,
    // Keeps track of already visited modules, in case a module re-exports its parent
    visited_mods: FxHashSet<DefId>,
}

impl<'a, 'tcx> LibEmbargoVisitor<'a, 'tcx> {
    crate fn new(cx: &'a mut crate::core::DocContext<'tcx>) -> LibEmbargoVisitor<'a, 'tcx> {
        LibEmbargoVisitor {
            tcx: cx.tcx,
            access_levels: &mut cx.cache.access_levels,
            prev_level: Some(AccessLevel::Public),
            visited_mods: FxHashSet::default(),
        }
    }

    crate fn visit_lib(&mut self, cnum: CrateNum) {
        let did = DefId { krate: cnum, index: CRATE_DEF_INDEX };
        self.update(did, Some(AccessLevel::Public));
        self.visit_mod(did);
    }

    // Updates node level and returns the updated level
    fn update(&mut self, did: DefId, level: Option<AccessLevel>) -> Option<AccessLevel> {
        let is_hidden = self.tcx.get_attrs(did).lists(sym::doc).has_word(sym::hidden);

        let old_level = self.access_levels.map.get(&did).cloned();
        // Accessibility levels can only grow
        if level > old_level && !is_hidden {
            self.access_levels.map.insert(did, level.unwrap());
            level
        } else {
            old_level
        }
    }

    crate fn visit_mod(&mut self, def_id: DefId) {
        if !self.visited_mods.insert(def_id) {
            return;
        }

        for item in self.tcx.item_children(def_id).iter() {
            if let Some(def_id) = item.res.opt_def_id() {
                if self.tcx.def_key(def_id).parent.map_or(false, |d| d == def_id.index)
                    || item.vis.is_public()
                {
                    self.visit_item(item.res);
                }
            }
        }
    }

    fn visit_item(&mut self, res: Res<!>) {
        let def_id = res.def_id();
        let vis = self.tcx.visibility(def_id);
        let inherited_item_level = if vis.is_public() { self.prev_level } else { None };

        let item_level = self.update(def_id, inherited_item_level);

        if let Res::Def(DefKind::Mod, _) = res {
            let orig_level = self.prev_level;

            self.prev_level = item_level;
            self.visit_mod(def_id);
            self.prev_level = orig_level;
        }
    }
}
