use std::mem;

use rustc_ast::visit::Visitor;
use rustc_ast::{Crate, EnumDef, ast, visit};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{CRATE_DEF_ID, LocalDefId};
use rustc_middle::middle::privacy::{EffectiveVisibilities, EffectiveVisibility, Level};
use rustc_middle::ty::Visibility;
use tracing::info;

use crate::{Decl, DeclKind, Resolver};

#[derive(Clone, Copy)]
enum UseChainId<'ra> {
    Def(LocalDefId),
    Import(Decl<'ra>),
}

trait Id<'a, 'ra, 'tcx> {
    fn level(&self) -> Level;
    fn effective_vis_or_private(
        self,
        visitor: &mut EffectiveVisibilitiesVisitor<'a, 'ra, 'tcx>,
    ) -> EffectiveVisibility;
    fn may_update(
        self,
        nominal_vis: Visibility,
        visitor: &EffectiveVisibilitiesVisitor<'a, 'ra, 'tcx>,
    ) -> Option<Option<Visibility>>;
}

impl<'a, 'ra, 'tcx> Id<'a, 'ra, 'tcx> for UseChainId<'ra> {
    fn level(&self) -> Level {
        match self {
            UseChainId::Def(_) => Level::Direct,
            UseChainId::Import(_) => Level::Reexported,
        }
    }

    fn effective_vis_or_private(
        self,
        visitor: &mut EffectiveVisibilitiesVisitor<'a, 'ra, 'tcx>,
    ) -> EffectiveVisibility {
        // Private nodes are only added to the table for caching, they could be added or removed at
        // any moment without consequences, so we don't set `changed` to true when adding them.
        match self {
            UseChainId::Def(def_id) => Id::effective_vis_or_private(def_id, visitor),
            UseChainId::Import(binding) => *visitor
                .import_effective_visibilities
                .effective_vis_or_private(binding, || visitor.r.private_vis_import(binding)),
        }
    }

    fn may_update(
        self,
        nominal_vis: Visibility,
        visitor: &EffectiveVisibilitiesVisitor<'a, 'ra, 'tcx>,
    ) -> Option<Option<Visibility>> {
        match self {
            UseChainId::Def(def_id) => Id::may_update(def_id, nominal_vis, visitor),
            UseChainId::Import(_) => Some(None),
        }
    }
}

impl<'a, 'ra, 'tcx> Id<'a, 'ra, 'tcx> for LocalDefId {
    fn level(&self) -> Level {
        Level::Direct
    }

    fn effective_vis_or_private(
        self,
        visitor: &mut EffectiveVisibilitiesVisitor<'a, 'ra, 'tcx>,
    ) -> EffectiveVisibility {
        *visitor
            .def_effective_visibilities
            .effective_vis_or_private(self, || visitor.r.private_vis_def(self))
    }

    fn may_update(
        self,
        nominal_vis: Visibility,
        visitor: &EffectiveVisibilitiesVisitor<'a, 'ra, 'tcx>,
    ) -> Option<Option<Visibility>> {
        (nominal_vis != visitor.current_private_vis
            && visitor.r.tcx.local_visibility(self) != visitor.current_private_vis)
            .then_some(Some(visitor.current_private_vis))
    }
}

trait EffectiveVisCollector<'a, 'ra: 'a, 'tcx: 'a> {
    type ParentId: Copy + Id<'a, 'ra, 'tcx>;

    fn base(&mut self) -> &mut EffectiveVisibilitiesVisitor<'a, 'ra, 'tcx>;

    /// All effective visibilities for a node are larger or equal than private visibility
    /// for that node (see `check_invariants` in middle/privacy.rs).
    /// So if either parent or nominal visibility is the same as private visibility, then
    /// `min(parent_vis, nominal_vis) <= private_vis`, and the update logic is guaranteed
    /// to not update anything and we can skip it.
    ///
    /// We are checking this condition only if the correct value of private visibility is
    /// cheaply available, otherwise it doesn't make sense performance-wise.
    ///
    /// `None` is returned if the update can be skipped,
    /// and cheap private visibility is returned otherwise.
    fn may_update(
        &mut self,
        nominal_vis: Visibility,
        parent: Self::ParentId,
    ) -> Option<Option<Visibility>> {
        Id::may_update(parent, nominal_vis, self.base())
    }

    #[must_use]
    fn update_def(
        &mut self,
        def_id: LocalDefId,
        nominal_vis: Visibility,
        parent_id: Self::ParentId,
    ) -> bool {
        let Some(cheap_private_vis) = self.may_update(nominal_vis, parent_id) else {
            return false;
        };
        let inherited_eff_vis = self.effective_vis_or_private(parent_id);
        let base = self.base();
        base.def_effective_visibilities.update(
            def_id,
            Some(nominal_vis),
            || cheap_private_vis.unwrap_or_else(|| base.r.private_vis_def(def_id)),
            inherited_eff_vis,
            parent_id.level(),
            base.r.tcx,
        )
    }

    #[must_use]
    fn update_import(&mut self, decl: Decl<'ra>, parent_id: Self::ParentId) -> bool {
        let nominal_vis = decl.vis().expect_local();
        let Some(cheap_private_vis) = self.may_update(nominal_vis, parent_id) else {
            return false;
        };
        let inherited_eff_vis = self.effective_vis_or_private(parent_id);
        let base = self.base();
        let changed = base.import_effective_visibilities.update(
            decl,
            Some(nominal_vis),
            || cheap_private_vis.unwrap_or_else(|| base.r.private_vis_import(decl)),
            inherited_eff_vis,
            parent_id.level(),
            base.r.tcx,
        );
        if let Some(max_vis_decl) = decl.ambiguity_vis_max.get() {
            // Avoid the most visible import in an ambiguous glob set being reported as unused.
            let _ = self.update_import(max_vis_decl, parent_id);
        }
        changed
    }

    fn effective_vis_or_private(&mut self, parent_id: Self::ParentId) -> EffectiveVisibility {
        Id::effective_vis_or_private(parent_id, self.base())
    }
}

pub(crate) struct EffectiveVisibilitiesVisitor<'a, 'ra, 'tcx> {
    r: &'a mut Resolver<'ra, 'tcx>,
    def_effective_visibilities: EffectiveVisibilities,
    /// While walking import chains we need to track effective visibilities per-decl, and def id
    /// keys in `Resolver::effective_visibilities` are not enough for that, because multiple
    /// declarations can correspond to a single def id in imports. So we keep a separate table.
    import_effective_visibilities: EffectiveVisibilities<Decl<'ra>>,
    // It's possible to recalculate this at any point, but it's relatively expensive.
    current_private_vis: Visibility,
}

impl Resolver<'_, '_> {
    fn nearest_normal_mod(&self, def_id: LocalDefId) -> LocalDefId {
        self.get_nearest_non_block_module(def_id.to_def_id()).nearest_parent_mod().expect_local()
    }

    fn private_vis_import(&self, decl: Decl<'_>) -> Visibility {
        let DeclKind::Import { import, .. } = decl.kind else { unreachable!() };
        Visibility::Restricted(
            import.def_id().map(|id| self.nearest_normal_mod(id)).unwrap_or(CRATE_DEF_ID),
        )
    }

    fn private_vis_def(&self, def_id: LocalDefId) -> Visibility {
        // For mod items `nearest_normal_mod` returns its argument, but we actually need its parent.
        let normal_mod_id = self.nearest_normal_mod(def_id);
        if normal_mod_id == def_id {
            Visibility::Restricted(self.tcx.local_parent(def_id))
        } else {
            Visibility::Restricted(normal_mod_id)
        }
    }
}

impl<'a, 'ra, 'tcx> EffectiveVisibilitiesVisitor<'a, 'ra, 'tcx> {
    /// Fills the `Resolver::effective_visibilities` table with public & exported items
    /// For now, this doesn't resolve macros (FIXME) and cannot resolve Impl, as we
    /// need access to a TyCtxt for that. Returns the set of ambiguous re-exports.
    pub(crate) fn compute_effective_visibilities<'c>(
        r: &'a mut Resolver<'ra, 'tcx>,
        krate: &'c Crate,
    ) -> FxHashSet<Decl<'ra>> {
        let mut visitor = EffectiveVisibilitiesVisitor {
            r,
            def_effective_visibilities: Default::default(),
            import_effective_visibilities: Default::default(),
            current_private_vis: Visibility::Restricted(CRATE_DEF_ID),
        };

        visitor.def_effective_visibilities.update_root();

        let mut chain_visitor = UseChainVisitor { visitor, queue: vec![] };
        chain_visitor.compute_effective_visibilities();

        let mut def_visitor = DefsVisitor { visitor: chain_visitor.visitor };
        def_visitor.update_bindings(CRATE_DEF_ID);
        visit::walk_crate(&mut def_visitor, krate);

        let visitor = def_visitor.visitor;
        visitor.r.effective_visibilities = visitor.def_effective_visibilities;

        let mut exported_ambiguities = FxHashSet::default();

        // Update visibilities for import def ids. These are not used during the
        // `EffectiveVisibilitiesVisitor` pass, because we have more detailed declaration-based
        // information, but are used by later passes. Effective visibility of an import def id
        // is the maximum value among visibilities of declarations corresponding to that def id.
        for (decl, eff_vis) in visitor.import_effective_visibilities.iter() {
            let DeclKind::Import { import, .. } = decl.kind else { unreachable!() };
            if let Some(def_id) = import.def_id() {
                r.effective_visibilities.update_eff_vis(def_id, eff_vis, r.tcx)
            }
            if decl.ambiguity.get().is_some() && eff_vis.is_public_at_level(Level::Reexported) {
                exported_ambiguities.insert(*decl);
            }
        }

        info!("resolve::effective_visibilities: {:#?}", r.effective_visibilities);

        exported_ambiguities
    }
}

#[derive(Copy, Clone)]
struct UpdateStep<'ra> {
    parent_mod: LocalDefId,
    binding: Decl<'ra>,
}

impl<'ra> UpdateStep<'ra> {
    fn new(parent_mod: LocalDefId, binding: Decl<'ra>) -> UpdateStep<'ra> {
        UpdateStep { parent_mod, binding }
    }
}

struct UseChainVisitor<'a, 'ra, 'tcx> {
    visitor: EffectiveVisibilitiesVisitor<'a, 'ra, 'tcx>,
    queue: Vec<UpdateStep<'ra>>,
}

impl<'a, 'ra, 'tcx> UseChainVisitor<'a, 'ra, 'tcx> {
    fn compute_effective_visibilities(&mut self) {
        self.collect_module_bindings(CRATE_DEF_ID);

        while let Some(UpdateStep { binding, parent_mod }) = self.queue.pop() {
            self.visitor.current_private_vis = Visibility::Restricted(parent_mod);
            self.update_binding_effective_visibility(parent_mod, binding);

            if !binding.is_import()
                && let Res::Def(DefKind::Mod, module_id) = binding.res()
                && let Some(module_id) = module_id.as_local()
            {
                self.collect_module_bindings(module_id);
            }
        }
    }

    fn try_append_binding(&mut self, binding: Decl<'ra>) {
        if (binding.is_import() || matches!(binding.res(), Res::Def(DefKind::Mod, _)))
            && let Some(parent_mod) = binding.parent_module
            && let Some(parent_mod) = parent_mod.opt_def_id()
            && let Some(parent_mod) = parent_mod.as_local()
        {
            self.queue.push(UpdateStep::new(parent_mod, binding));
        }
    }

    /// Update effective visibility of a name declaration in the given module,
    /// including its whole reexport chain.
    fn update_binding_effective_visibility(&mut self, parent_mod: LocalDefId, mut decl: Decl<'ra>) {
        let mut parent_id = UseChainId::Def(parent_mod);
        while let DeclKind::Import { source_decl, import: _ } = decl.kind {
            let _ = self.update_import(decl, parent_id);
            parent_id = UseChainId::Import(decl);
            decl = source_decl;
        }

        if let Some(def_id) = decl.res().opt_def_id().and_then(|id| id.as_local()) {
            if self.update_def(def_id, decl.vis().expect_local(), parent_id) {
                self.try_append_binding(decl);
            }
        }
    }

    fn collect_module_bindings(&mut self, module_id: LocalDefId) {
        let module = self.visitor.r.expect_module(module_id.to_def_id());
        for (_, name_resolution) in self.visitor.r.resolutions(module).borrow().iter() {
            let Some(decl) = name_resolution.borrow().best_decl() else {
                continue;
            };
            self.try_append_binding(decl);
        }
    }
}

impl<'a, 'ra, 'tcx> EffectiveVisCollector<'a, 'ra, 'tcx> for UseChainVisitor<'a, 'ra, 'tcx> {
    type ParentId = UseChainId<'ra>;

    fn base(&mut self) -> &mut EffectiveVisibilitiesVisitor<'a, 'ra, 'tcx> {
        &mut self.visitor
    }
}

struct DefsVisitor<'a, 'ra, 'tcx> {
    visitor: EffectiveVisibilitiesVisitor<'a, 'ra, 'tcx>,
}

impl<'a, 'ra, 'tcx> DefsVisitor<'a, 'ra, 'tcx> {
    fn update_bindings(&mut self, module_id: LocalDefId) {
        let module = self.base().r.expect_module(module_id.to_def_id());
        for (_, name_resolution) in self.base().r.resolutions(module).borrow().iter() {
            let Some(decl) = name_resolution.borrow().best_decl() else {
                continue;
            };
            if !decl.is_import()
                && let Some(def_id) = decl.res().opt_def_id().and_then(|id| id.as_local())
            {
                _ = self.update_def(def_id, decl.vis().expect_local(), module_id);
            }
        }
    }
}

impl<'a, 'ra, 'tcx> DefsVisitor<'a, 'ra, 'tcx> {
    fn update_field(&mut self, def_id: LocalDefId, parent_id: LocalDefId) {
        let vis = self.base().r.tcx.local_visibility(def_id);
        _ = self.update_def(def_id, vis, parent_id);
    }
}

impl<'a, 'ra, 'tcx> EffectiveVisCollector<'a, 'ra, 'tcx> for DefsVisitor<'a, 'ra, 'tcx> {
    type ParentId = LocalDefId;

    fn base(&mut self) -> &mut EffectiveVisibilitiesVisitor<'a, 'ra, 'tcx> {
        &mut self.visitor
    }

    fn update_import(&mut self, _decl: Decl<'ra>, _parent_id: Self::ParentId) -> bool {
        unreachable!()
    }
}

impl<'a, 'ra, 'tcx> Visitor<'a> for DefsVisitor<'a, 'ra, 'tcx> {
    fn visit_item(&mut self, item: &'a ast::Item) {
        let def_id = self.base().r.local_def_id(item.id);
        // Update effective visibilities of nested items.
        // If it's a mod, also make the visitor walk all of its items
        match item.kind {
            // Resolved in rustc_privacy when types are available
            ast::ItemKind::Impl(..) => return,

            // Should be unreachable at this stage
            ast::ItemKind::MacCall(..) | ast::ItemKind::DelegationMac(..) => panic!(
                "ast::ItemKind::MacCall encountered, this should not anymore appear at this stage"
            ),

            ast::ItemKind::Mod(..) => {
                let prev_private_vis = mem::replace(
                    &mut self.base().current_private_vis,
                    Visibility::Restricted(def_id),
                );
                self.update_bindings(def_id);
                visit::walk_item(self, item);
                self.base().current_private_vis = prev_private_vis;
            }

            ast::ItemKind::Enum(_, _, EnumDef { ref variants }) => {
                self.update_bindings(def_id);
                for variant in variants {
                    let variant_def_id = self.base().r.local_def_id(variant.id);
                    for field in variant.data.fields() {
                        let field_def_id = self.base().r.local_def_id(field.id);
                        self.update_field(field_def_id, variant_def_id);
                    }
                }
            }

            ast::ItemKind::Struct(_, _, ref def) | ast::ItemKind::Union(_, _, ref def) => {
                for field in def.fields() {
                    let field_def_id = self.base().r.local_def_id(field.id);
                    self.update_field(field_def_id, def_id);
                }
            }

            ast::ItemKind::Trait(..) => {
                self.update_bindings(def_id);
            }

            ast::ItemKind::ExternCrate(..)
            | ast::ItemKind::Use(..)
            | ast::ItemKind::Static(..)
            | ast::ItemKind::Const(..)
            | ast::ItemKind::ConstBlock(..)
            | ast::ItemKind::GlobalAsm(..)
            | ast::ItemKind::TyAlias(..)
            | ast::ItemKind::TraitAlias(..)
            | ast::ItemKind::MacroDef(..)
            | ast::ItemKind::ForeignMod(..)
            | ast::ItemKind::Fn(..)
            | ast::ItemKind::Delegation(..) => return,
        }
    }
}
