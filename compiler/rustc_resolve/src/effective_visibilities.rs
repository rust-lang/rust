use std::mem;

use rustc_ast::visit::Visitor;
use rustc_ast::{Attribute, Crate, EnumDef, ast, visit};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{CRATE_DEF_ID, LocalDefId};
use rustc_middle::middle::privacy::{EffectiveVisibilities, EffectiveVisibility, Level};
use rustc_middle::ty::Visibility;
use rustc_span::sym;
use tracing::info;

use crate::{Decl, DeclKind, Resolver};

#[derive(Clone, Copy)]
enum ParentId<'ra> {
    Def(LocalDefId),
    Import(Decl<'ra>),
}

impl ParentId<'_> {
    fn level(self) -> Level {
        match self {
            ParentId::Def(_) => Level::Direct,
            ParentId::Import(_) => Level::Reexported,
        }
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
    /// A set of pairs corresponding to modules, where the first module is
    /// reachable via a macro that's defined in the second module. This cannot
    /// be represented as reachable because it can't handle the following case:
    ///
    /// pub mod n {                         // Should be `Public`
    ///     pub(crate) mod p {              // Should *not* be accessible
    ///         pub fn f() -> i32 { 12 }    // Must be `Reachable`
    ///     }
    /// }
    /// pub macro m() {
    ///     n::p::f()
    /// }
    macro_reachable: FxHashSet<(LocalDefId, LocalDefId)>,
    changed: bool,
}

impl Resolver<'_, '_> {
    fn private_vis_decl(&self, decl: Decl<'_>) -> Visibility {
        Visibility::Restricted(
            decl.parent_module.map_or(CRATE_DEF_ID, |m| m.nearest_parent_mod().expect_local()),
        )
    }

    fn private_vis_def(&self, def_id: LocalDefId) -> Visibility {
        // For mod items `normal_mod_id` will be equal to `def_id`, but we actually need its parent.
        let normal_mod_id = self
            .get_nearest_non_block_module(def_id.to_def_id())
            .nearest_parent_mod()
            .expect_local();
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
            macro_reachable: Default::default(),
            changed: true,
        };

        visitor.def_effective_visibilities.update_root();
        visitor.set_bindings_effective_visibilities(CRATE_DEF_ID);

        while visitor.changed {
            visitor.changed = false;
            visit::walk_crate(&mut visitor, krate);
        }
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

    /// Update effective visibilities of name declarations in the given module,
    /// including their whole reexport chains.
    fn set_bindings_effective_visibilities(&mut self, module_id: LocalDefId) {
        let module = self.r.expect_module(module_id.to_def_id());
        for (_, name_resolution) in self.r.resolutions(module).borrow().iter() {
            let Some(mut decl) = name_resolution.borrow().best_decl() else {
                continue;
            };
            // Set the given effective visibility level to `Level::Direct` and
            // sets the rest of the `use` chain to `Level::Reexported` until
            // we hit the actual exported item.
            let priv_vis = |this: &Self, parent_id, decl| match parent_id {
                ParentId::Def(_) => this.current_private_vis,
                ParentId::Import(_) => this.r.private_vis_decl(decl),
            };
            let mut parent_id = ParentId::Def(module_id);
            while let DeclKind::Import { source_decl, .. } = decl.kind {
                self.update_import(decl, parent_id, priv_vis(self, parent_id, decl));
                parent_id = ParentId::Import(decl);
                decl = source_decl;
            }
            if let Some(def_id) = decl.res().opt_def_id().and_then(|id| id.as_local()) {
                let priv_vis = priv_vis(self, parent_id, decl);
                self.update_def(def_id, decl.vis().expect_local(), parent_id, priv_vis);
            }
        }
    }

    fn effective_vis_or_private(&mut self, parent_id: ParentId<'ra>) -> EffectiveVisibility {
        // Private nodes are only added to the table for caching, they could be added or removed at
        // any moment without consequences, so we don't set `changed` to true when adding them.
        *match parent_id {
            ParentId::Def(def_id) => self
                .def_effective_visibilities
                .effective_vis_or_private(def_id, || self.r.private_vis_def(def_id)),
            ParentId::Import(binding) => self
                .import_effective_visibilities
                .effective_vis_or_private(binding, || self.r.private_vis_decl(binding)),
        }
    }

    /// All effective visibilities for a node are larger or equal than private visibility
    /// for that node (see `check_invariants` in middle/privacy.rs).
    /// So if either parent or nominal visibility is the same as private visibility, then
    /// `min(parent_vis, nominal_vis) <= priv_vis`, and the update logic is guaranteed
    /// to not update anything and we can skip it.
    fn may_update(
        &self,
        nominal_vis: Visibility,
        parent_id: ParentId<'_>,
        priv_vis: Visibility,
    ) -> bool {
        nominal_vis != priv_vis
            && match parent_id {
                ParentId::Def(def_id) => self.r.tcx.local_visibility(def_id),
                ParentId::Import(decl) => decl.vis().expect_local(),
            } != priv_vis
    }

    fn update_import(&mut self, decl: Decl<'ra>, parent_id: ParentId<'ra>, priv_vis: Visibility) {
        let nominal_vis = decl.vis().expect_local();
        if !self.may_update(nominal_vis, parent_id, priv_vis) {
            return;
        };
        let inherited_eff_vis = self.effective_vis_or_private(parent_id);
        let tcx = self.r.tcx;
        self.changed |= self.import_effective_visibilities.update(
            decl,
            Some(nominal_vis),
            priv_vis,
            inherited_eff_vis,
            parent_id.level(),
            tcx,
        );
        if let Some(max_vis_decl) = decl.ambiguity_vis_max.get() {
            // Avoid the most visible import in an ambiguous glob set being reported as unused.
            self.update_import(max_vis_decl, parent_id, priv_vis);
        }
    }

    fn update_def(
        &mut self,
        def_id: LocalDefId,
        nominal_vis: Visibility,
        parent_id: ParentId<'ra>,
        priv_vis: Visibility,
    ) {
        if !self.may_update(nominal_vis, parent_id, priv_vis) {
            return;
        };
        let inherited_eff_vis = self.effective_vis_or_private(parent_id);
        let tcx = self.r.tcx;
        self.changed |= self.def_effective_visibilities.update(
            def_id,
            Some(nominal_vis),
            priv_vis,
            inherited_eff_vis,
            parent_id.level(),
            tcx,
        );
    }

    fn update_field(&mut self, def_id: LocalDefId, parent_id: LocalDefId) {
        let nominal_vis = self.r.tcx.local_visibility(def_id);
        self.update_def(def_id, nominal_vis, ParentId::Def(parent_id), self.current_private_vis);
    }

    fn update_macro(&mut self, def_id: LocalDefId, inherited_effective_vis: EffectiveVisibility) {
        let max_vis = Some(self.r.tcx.local_visibility(def_id));
        let priv_vis = if def_id == CRATE_DEF_ID {
            Visibility::Restricted(CRATE_DEF_ID)
        } else {
            self.r.private_vis_def(def_id)
        };
        self.changed |= self.def_effective_visibilities.update(
            def_id,
            max_vis,
            priv_vis,
            inherited_effective_vis,
            Level::Reachable,
            self.r.tcx,
        );
    }

    // We have to make sure that the items that macros might reference
    // are reachable, since they might be exported transitively.
    fn update_reachability_from_macro(
        &mut self,
        local_def_id: LocalDefId,
        md: &ast::MacroDef,
        attrs: &[Attribute],
    ) {
        // Non-opaque macros cannot make other items more accessible than they already are.
        if rustc_ast::attr::find_by_name(attrs, sym::rustc_macro_transparency)
            .map_or(md.macro_rules, |attr| attr.value_str() != Some(sym::opaque))
        {
            return;
        }

        let macro_module_def_id = self.r.tcx.local_parent(local_def_id);
        if self.r.tcx.def_kind(macro_module_def_id) != DefKind::Mod {
            // The macro's parent doesn't correspond to a `mod`, return early (#63164, #65252).
            return;
        }

        let Some(macro_ev) = self
            .def_effective_visibilities
            .effective_vis(local_def_id)
            .filter(|ev| ev.public_at_level().is_some())
            .copied()
        else {
            return;
        };

        // Since we are starting from an externally visible module,
        // all the parents in the loop below are also guaranteed to be modules.
        let mut module_def_id = macro_module_def_id;
        loop {
            let changed_reachability =
                self.update_macro_reachable(module_def_id, macro_module_def_id, macro_ev);
            if changed_reachability || module_def_id == CRATE_DEF_ID {
                break;
            }
            module_def_id = self.r.tcx.local_parent(module_def_id);
        }
    }

    /// Updates the item as being reachable through a macro defined in the given
    /// module. Returns `true` if the level has changed.
    fn update_macro_reachable(
        &mut self,
        module_def_id: LocalDefId,
        defining_mod: LocalDefId,
        macro_ev: EffectiveVisibility,
    ) -> bool {
        if self.macro_reachable.insert((module_def_id, defining_mod)) {
            let module = self.r.expect_module(module_def_id.to_def_id());
            for (_, name_resolution) in self.r.resolutions(module).borrow().iter() {
                let Some(decl) = name_resolution.borrow().best_decl() else {
                    continue;
                };

                if let Res::Def(def_kind, def_id) = decl.res()
                    && let Some(def_id) = def_id.as_local()
                    // FIXME: defs should be checked with `EffectiveVisibilities::is_reachable`.
                    && decl.vis().is_accessible_from(defining_mod, self.r.tcx)
                {
                    let vis = self.r.tcx.local_visibility(def_id);
                    self.update_macro_reachable_def(def_id, def_kind, vis, defining_mod, macro_ev);
                }
            }
            true
        } else {
            false
        }
    }

    fn update_macro_reachable_def(
        &mut self,
        def_id: LocalDefId,
        def_kind: DefKind,
        vis: Visibility,
        module: LocalDefId,
        macro_ev: EffectiveVisibility,
    ) {
        self.update_macro(def_id, macro_ev);

        match def_kind {
            DefKind::Mod => {
                if vis.is_accessible_from(module, self.r.tcx) {
                    self.update_macro_reachable(def_id, module, macro_ev);
                }
            }
            DefKind::Struct | DefKind::Union => {
                self.r
                    .macro_reachable_adts
                    .entry(def_id)
                    .or_insert_with(Default::default)
                    .insert(module);
            }
            _ => {}
        }
    }
}

impl<'a, 'ra, 'tcx> Visitor<'a> for EffectiveVisibilitiesVisitor<'a, 'ra, 'tcx> {
    fn visit_item(&mut self, item: &'a ast::Item) {
        let def_id = self.r.local_def_id(item.id);
        // Update effective visibilities of nested items.
        // If it's a mod, also make the visitor walk all of its items
        match &item.kind {
            // Resolved in rustc_privacy when types are available
            ast::ItemKind::Impl(..) => return,

            // Should be unreachable at this stage
            ast::ItemKind::MacCall(..) | ast::ItemKind::DelegationMac(..) => panic!(
                "ast::ItemKind::MacCall encountered, this should not anymore appear at this stage"
            ),

            ast::ItemKind::Mod(..) => {
                let prev_private_vis =
                    mem::replace(&mut self.current_private_vis, Visibility::Restricted(def_id));
                self.set_bindings_effective_visibilities(def_id);
                visit::walk_item(self, item);
                self.current_private_vis = prev_private_vis;
            }

            ast::ItemKind::Enum(_, _, EnumDef { variants }) => {
                self.set_bindings_effective_visibilities(def_id);
                for variant in variants {
                    let variant_def_id = self.r.local_def_id(variant.id);
                    for field in variant.data.fields() {
                        self.update_field(self.r.local_def_id(field.id), variant_def_id);
                    }
                }
            }

            ast::ItemKind::Struct(_, _, def) | ast::ItemKind::Union(_, _, def) => {
                for field in def.fields() {
                    self.update_field(self.r.local_def_id(field.id), def_id);
                }
            }

            ast::ItemKind::Trait(..) => {
                self.set_bindings_effective_visibilities(def_id);
            }

            ast::ItemKind::MacroDef(_, macro_def) => {
                self.update_reachability_from_macro(def_id, macro_def, &item.attrs);
            }

            ast::ItemKind::ExternCrate(..)
            | ast::ItemKind::Use(..)
            | ast::ItemKind::Static(..)
            | ast::ItemKind::Const(..)
            | ast::ItemKind::ConstBlock(..)
            | ast::ItemKind::GlobalAsm(..)
            | ast::ItemKind::TyAlias(..)
            | ast::ItemKind::TraitAlias(..)
            | ast::ItemKind::ForeignMod(..)
            | ast::ItemKind::Fn(..)
            | ast::ItemKind::Delegation(..) => return,
        }
    }
}
