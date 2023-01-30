use crate::core::DocContext;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, DefIdSet};
use rustc_middle::ty::TyCtxt;
use rustc_resolve::Resolver;

// FIXME: this may not be exhaustive, but is sufficient for rustdocs current uses

#[derive(Default)]
pub(crate) struct RustdocEffectiveVisibilities {
    extern_public: DefIdSet,
}

macro_rules! define_method {
    ($method:ident) => {
        pub(crate) fn $method(&self, tcx: TyCtxt<'_>, def_id: DefId) -> bool {
            match def_id.as_local() {
                Some(def_id) => tcx.effective_visibilities(()).$method(def_id),
                None => self.extern_public.contains(&def_id),
            }
        }
    };
}

impl RustdocEffectiveVisibilities {
    define_method!(is_directly_public);
    define_method!(is_exported);
    define_method!(is_reachable);

    pub(crate) fn init(&mut self, extern_public: DefIdSet) {
        self.extern_public = extern_public;
    }
}

pub(crate) fn lib_embargo_visit_item(cx: &mut DocContext<'_>, def_id: DefId) {
    assert!(!def_id.is_local());
    LibEmbargoVisitor {
        tcx: cx.tcx,
        extern_public: &mut cx.cache.effective_visibilities.extern_public,
        visited_mods: Default::default(),
    }
    .visit_item(def_id)
}

pub(crate) fn early_lib_embargo_visit_item(
    resolver: &Resolver<'_>,
    extern_public: &mut DefIdSet,
    def_id: DefId,
    is_mod: bool,
) {
    assert!(!def_id.is_local());
    EarlyLibEmbargoVisitor { resolver, extern_public, visited_mods: Default::default() }
        .visit_item(def_id, is_mod)
}

/// Similar to `librustc_privacy::EmbargoVisitor`, but also takes
/// specific rustdoc annotations into account (i.e., `doc(hidden)`)
struct LibEmbargoVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    // Effective visibilities for reachable nodes
    extern_public: &'a mut DefIdSet,
    // Keeps track of already visited modules, in case a module re-exports its parent
    visited_mods: DefIdSet,
}

struct EarlyLibEmbargoVisitor<'r, 'ra> {
    resolver: &'r Resolver<'ra>,
    // Effective visibilities for reachable nodes
    extern_public: &'r mut DefIdSet,
    // Keeps track of already visited modules, in case a module re-exports its parent
    visited_mods: DefIdSet,
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
            self.extern_public.insert(def_id);
            if self.tcx.def_kind(def_id) == DefKind::Mod {
                self.visit_mod(def_id);
            }
        }
    }
}

impl EarlyLibEmbargoVisitor<'_, '_> {
    fn visit_mod(&mut self, def_id: DefId) {
        if !self.visited_mods.insert(def_id) {
            return;
        }

        for item in self.resolver.cstore().module_children_untracked(def_id, self.resolver.sess()) {
            if let Some(def_id) = item.res.opt_def_id() {
                if item.vis.is_public() {
                    self.visit_item(def_id, matches!(item.res, Res::Def(DefKind::Mod, _)));
                }
            }
        }
    }

    fn visit_item(&mut self, def_id: DefId, is_mod: bool) {
        if !self.resolver.cstore().is_doc_hidden_untracked(def_id) {
            self.extern_public.insert(def_id);
            if is_mod {
                self.visit_mod(def_id);
            }
        }
    }
}
