use std::sync::RwLock;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{CrateNum, DefId, DefIdSet};
use rustc_middle::ty::TyCtxt;

use crate::core::DocContext;

// FIXME: this may not be exhaustive, but is sufficient for rustdocs current uses

#[derive(Default)]
pub(crate) struct RustdocEffectiveVisibilities {
    inner: RwLock<VisInner>,
    pub(crate) document_hidden: bool,
}

#[derive(Default)]
struct VisInner {
    visited: FxHashSet<CrateNum>,
    extern_public: DefIdSet,
}

macro_rules! define_method {
    ($method:ident) => {
        pub(crate) fn $method(&self, tcx: TyCtxt<'_>, def_id: DefId) -> bool {
            match def_id.as_local() {
                Some(def_id) => tcx.effective_visibilities(()).$method(def_id),
                None => {
                    let vis = self.inner.read().unwrap();
                    if vis.visited.contains(&def_id.krate) {
                        vis.extern_public.contains(&def_id)
                    } else {
                        std::mem::drop(vis);
                        lib_embargo_visit_item_(&self, tcx, def_id.krate.as_def_id());
                        let mut vis = self.inner.write().unwrap();
                        vis.visited.insert(def_id.krate);
                        vis.extern_public.contains(&def_id)
                    }
                }
            }
        }
    };
}

impl RustdocEffectiveVisibilities {
    define_method!(is_directly_public);
    define_method!(is_exported);
    define_method!(is_reachable);
}

pub(crate) fn lib_embargo_visit_item_(
    vis: &RustdocEffectiveVisibilities,
    tcx: TyCtxt<'_>,
    def_id: DefId,
) {
    assert!(!def_id.is_local());
    LibEmbargoVisitor {
        tcx,
        extern_public: &mut vis.inner.write().unwrap().extern_public,
        visited_mods: Default::default(),
        document_hidden: vis.document_hidden,
    }
    .visit_item(def_id)
}

pub(crate) fn lib_embargo_visit_item(cx: &mut DocContext<'_>, def_id: DefId) {
    assert_eq!(cx.cache.effective_visibilities.document_hidden, cx.render_options.document_hidden);
    lib_embargo_visit_item_(&cx.cache.effective_visibilities, cx.tcx, def_id)
}

/// Similar to `librustc_privacy::EmbargoVisitor`, but also takes
/// specific rustdoc annotations into account (i.e., `doc(hidden)`)
struct LibEmbargoVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    // Effective visibilities for reachable nodes
    extern_public: &'a mut DefIdSet,
    // Keeps track of already visited modules, in case a module re-exports its parent
    visited_mods: DefIdSet,
    document_hidden: bool,
}

impl LibEmbargoVisitor<'_, '_> {
    fn visit_mod(&mut self, def_id: DefId) {
        if !self.visited_mods.insert(def_id) {
            return;
        }

        for item in self.tcx.module_children(def_id).iter() {
            if let Some(def_id) = item.res.opt_def_id()
                && item.vis.is_public()
            {
                self.visit_item(def_id);
            }
        }
    }

    fn visit_item(&mut self, def_id: DefId) {
        if self.document_hidden || !self.tcx.is_doc_hidden(def_id) {
            self.extern_public.insert(def_id);
            if self.tcx.def_kind(def_id) == DefKind::Mod {
                self.visit_mod(def_id);
            }
        }
    }
}
