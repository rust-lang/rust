use crate::{ImportKind, NameBindingKind, Resolver};
use rustc_ast::ast;
use rustc_ast::visit;
use rustc_ast::visit::Visitor;
use rustc_ast::Crate;
use rustc_ast::EnumDef;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::def_id::CRATE_DEF_ID;
use rustc_middle::middle::privacy::AccessLevel;
use rustc_middle::ty::{DefIdTree, Visibility};

pub struct AccessLevelsVisitor<'r, 'a> {
    r: &'r mut Resolver<'a>,
    changed: bool,
}

impl<'r, 'a> AccessLevelsVisitor<'r, 'a> {
    /// Fills the `Resolver::access_levels` table with public & exported items
    /// For now, this doesn't resolve macros (FIXME) and cannot resolve Impl, as we
    /// need access to a TyCtxt for that.
    pub fn compute_access_levels<'c>(r: &'r mut Resolver<'a>, krate: &'c Crate) {
        let mut visitor = AccessLevelsVisitor { r, changed: false };

        visitor.update(CRATE_DEF_ID, Visibility::Public, CRATE_DEF_ID, AccessLevel::Public);
        visitor.set_bindings_access_level(CRATE_DEF_ID);

        while visitor.changed {
            visitor.reset();
            visit::walk_crate(&mut visitor, krate);
        }

        info!("resolve::access_levels: {:#?}", r.access_levels);
    }

    fn reset(&mut self) {
        self.changed = false;
    }

    /// Update the access level of the bindings in the given module accordingly. The module access
    /// level has to be Exported or Public.
    /// This will also follow `use` chains (see PrivacyVisitor::set_import_binding_access_level).
    fn set_bindings_access_level(&mut self, module_id: LocalDefId) {
        assert!(self.r.module_map.contains_key(&&module_id.to_def_id()));
        let module = self.r.get_module(module_id.to_def_id()).unwrap();
        let resolutions = self.r.resolutions(module);

        for (_, name_resolution) in resolutions.borrow().iter() {
            if let Some(mut binding) = name_resolution.borrow().binding() && !binding.is_ambiguity() {
                // Set the given binding access level to `AccessLevel::Public` and
                // sets the rest of the `use` chain to `AccessLevel::Exported` until
                // we hit the actual exported item.

                // FIXME: tag and is_public() condition should be removed, but assertions occur.
                let tag = if binding.is_import() { AccessLevel::Exported } else { AccessLevel::Public };
                if binding.vis.is_public() {
                    let mut prev_parent_id = module_id;
                    let mut level = AccessLevel::Public;
                    while let NameBindingKind::Import { binding: nested_binding, import, .. } =
                        binding.kind
                    {
                        let mut update = |node_id| self.update(
                            self.r.local_def_id(node_id),
                            binding.vis.expect_local(),
                            prev_parent_id,
                            level,
                        );
                        // In theory all the import IDs have individual visibilities and effective
                        // visibilities, but in practice these IDs go straigth to HIR where all
                        // their few uses assume that their (effective) visibility applies to the
                        // whole syntactic `use` item. So we update them all to the maximum value
                        // among the potential individual effective visibilities. Maybe HIR for
                        // imports shouldn't use three IDs at all.
                        update(import.id);
                        if let ImportKind::Single { additional_ids, .. } = import.kind {
                            update(additional_ids.0);
                            update(additional_ids.1);
                        }

                        level = AccessLevel::Exported;
                        prev_parent_id = self.r.local_def_id(import.id);
                        binding = nested_binding;
                    }
                }

                if let Some(def_id) = binding.res().opt_def_id().and_then(|id| id.as_local()) {
                    self.update(def_id, binding.vis.expect_local(), module_id, tag);
                }
            }
        }
    }

    fn update(
        &mut self,
        def_id: LocalDefId,
        nominal_vis: Visibility,
        parent_id: LocalDefId,
        tag: AccessLevel,
    ) {
        let mut access_levels = std::mem::take(&mut self.r.access_levels);
        let module_id =
            self.r.get_nearest_non_block_module(def_id.to_def_id()).def_id().expect_local();
        let res = access_levels.update(
            def_id,
            nominal_vis,
            || Visibility::Restricted(module_id),
            parent_id,
            tag,
            &*self.r,
        );
        if let Ok(changed) = res {
            self.changed |= changed;
        } else {
            self.r.session.delay_span_bug(
                self.r.opt_span(def_id.to_def_id()).unwrap(),
                "Can't update effective visibility",
            );
        }
        self.r.access_levels = access_levels;
    }
}

impl<'r, 'ast> Visitor<'ast> for AccessLevelsVisitor<'ast, 'r> {
    fn visit_item(&mut self, item: &'ast ast::Item) {
        let def_id = self.r.local_def_id(item.id);
        // Set access level of nested items.
        // If it's a mod, also make the visitor walk all of its items
        match item.kind {
            // Resolved in rustc_privacy when types are available
            ast::ItemKind::Impl(..) => return,

            // Should be unreachable at this stage
            ast::ItemKind::MacCall(..) => panic!(
                "ast::ItemKind::MacCall encountered, this should not anymore appear at this stage"
            ),

            // Foreign modules inherit level from parents.
            ast::ItemKind::ForeignMod(..) => {
                let parent_id = self.r.local_parent(def_id);
                self.update(def_id, Visibility::Public, parent_id, AccessLevel::Public);
            }

            // Only exported `macro_rules!` items are public, but they always are
            ast::ItemKind::MacroDef(ref macro_def) if macro_def.macro_rules => {
                let parent_id = self.r.local_parent(def_id);
                let vis = self.r.visibilities[&def_id];
                self.update(def_id, vis, parent_id, AccessLevel::Public);
            }

            ast::ItemKind::Mod(..) => {
                self.set_bindings_access_level(def_id);
                visit::walk_item(self, item);
            }

            ast::ItemKind::Enum(EnumDef { ref variants }, _) => {
                self.set_bindings_access_level(def_id);
                for variant in variants {
                    let variant_def_id = self.r.local_def_id(variant.id);
                    for field in variant.data.fields() {
                        let field_def_id = self.r.local_def_id(field.id);
                        let vis = self.r.visibilities[&field_def_id];
                        self.update(field_def_id, vis, variant_def_id, AccessLevel::Public);
                    }
                }
            }

            ast::ItemKind::Struct(ref def, _) | ast::ItemKind::Union(ref def, _) => {
                for field in def.fields() {
                    let field_def_id = self.r.local_def_id(field.id);
                    let vis = self.r.visibilities[&field_def_id];
                    self.update(field_def_id, vis, def_id, AccessLevel::Public);
                }
            }

            ast::ItemKind::Trait(..) => {
                self.set_bindings_access_level(def_id);
            }

            ast::ItemKind::ExternCrate(..)
            | ast::ItemKind::Use(..)
            | ast::ItemKind::Static(..)
            | ast::ItemKind::Const(..)
            | ast::ItemKind::GlobalAsm(..)
            | ast::ItemKind::TyAlias(..)
            | ast::ItemKind::TraitAlias(..)
            | ast::ItemKind::MacroDef(..)
            | ast::ItemKind::Fn(..) => return,
        }
    }
}
