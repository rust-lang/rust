use crate::core::DocContext;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, CRATE_DEF_ID};
use rustc_middle::middle::privacy::{EffectiveVisibilities, Level};
use rustc_middle::ty::{TyCtxt, Visibility};

// FIXME: this may not be exhaustive, but is sufficient for rustdocs current uses

pub(crate) fn lib_embargo_visit_item(cx: &mut DocContext<'_>, def_id: DefId) {
    assert!(!def_id.is_local());
    LibEmbargoVisitor {
        tcx: cx.tcx,
        effective_visibilities: &mut cx.cache.effective_visibilities,
        visited_mods: FxHashSet::default(),
    }
    .visit_item(def_id)
}

/// Similar to `librustc_privacy::EmbargoVisitor`, but also takes
/// specific rustdoc annotations into account (i.e., `doc(hidden)`)
struct LibEmbargoVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    // Effective visibilities for reachable nodes
    effective_visibilities: &'a mut EffectiveVisibilities<DefId>,
    // Keeps track of already visited modules, in case a module re-exports its parent
    visited_mods: FxHashSet<DefId>,
}

impl LibEmbargoVisitor<'_, '_> {
    fn visit_mod(&mut self, def_id: DefId) {
        if !self.visited_mods.insert(def_id) {
            return;
        }

        for item in self.tcx.module_children(def_id).iter() {
            if let Some(def_id) = item.res.opt_def_id() {
                if item.vis.is_public() {
                    self.visit_item(def_id);
                }
            }
        }
    }

    fn visit_item(&mut self, def_id: DefId) {
        if !self.tcx.is_doc_hidden(def_id) {
            self.effective_visibilities.set_public_at_level(
                def_id,
                || Visibility::Restricted(CRATE_DEF_ID),
                Level::Direct,
            );
            if self.tcx.def_kind(def_id) == DefKind::Mod {
                self.visit_mod(def_id);
            }
        }
    }
}
