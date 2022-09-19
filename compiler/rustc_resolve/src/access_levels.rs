use crate::imports::ImportKind;
use crate::NameBinding;
use crate::NameBindingKind;
use crate::Resolver;
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

        visitor.set_access_level_def_id(CRATE_DEF_ID, Some(AccessLevel::Public));
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

        // Set the given binding access level to `AccessLevel::Public` and
        // sets the rest of the `use` chain to `AccessLevel::Exported` until
        // we hit the actual exported item.
        let set_import_binding_access_level =
            |this: &mut Self, mut binding: &NameBinding<'a>, mut parent_id| {
                while let NameBindingKind::Import { binding: nested_binding, import, .. } =
                    binding.kind
                {
                    if this.r.opt_local_def_id(import.id).is_some() {
                        let vis = match binding.vis {
                            Visibility::Public => Visibility::Public,
                            Visibility::Restricted(id) => Visibility::Restricted(id.expect_local())
                        };
                        this.update_effective_vis(this.r.local_def_id(import.id), vis, parent_id, AccessLevel::Exported);
                        if let ImportKind::Single { additional_ids, .. } = import.kind {

                            if let Some(id) = this.r.opt_local_def_id(additional_ids.0) {
                                this.update_effective_vis(id, vis, parent_id, AccessLevel::Exported);
                            }

                            if let Some(id) = this.r.opt_local_def_id(additional_ids.1) {
                                this.update_effective_vis(id, vis, parent_id, AccessLevel::Exported);
                            }
                        }

                        parent_id = this.r.local_def_id(import.id);
                    }
                    binding = nested_binding;
                }
            };

            let module = self.r.get_module(module_id.to_def_id()).unwrap();
            let resolutions = self.r.resolutions(module);

        for (.., name_resolution) in resolutions.borrow().iter() {
            if let Some(binding) = name_resolution.borrow().binding() {
                let tag = match binding.is_import() {
                    true => {
                        if !binding.is_ambiguity() {
                            set_import_binding_access_level(self, binding, module_id);
                        }
                        AccessLevel::Exported
                    },
                    false => AccessLevel::Public
                };

                if let Some(def_id) = binding.res().opt_def_id().and_then(|id| id.as_local()) && !binding.is_ambiguity(){
                    let vis = match binding.vis {
                        Visibility::Public => Visibility::Public,
                        Visibility::Restricted(id) => Visibility::Restricted(id.expect_local())
                    };
                    self.update_effective_vis(def_id, vis, module_id, tag);
                }
            }
        }
    }

    fn update_effective_vis(
        &mut self,
        current_id: LocalDefId,
        current_vis: Visibility,
        module_id: LocalDefId,
        tag: AccessLevel,
    ) {
        if let Some(inherited_effective_vis) = self.r.access_levels.get_effective_vis(module_id) {
            let mut current_effective_vis = self.r.access_levels.get_effective_vis(current_id).copied().unwrap_or_default();
            let current_effective_vis_copy = current_effective_vis.clone();
            for level in [
                AccessLevel::Public,
                AccessLevel::Exported,
                AccessLevel::Reachable,
                AccessLevel::ReachableFromImplTrait,
            ] {
                if level <= tag {
                    let nearest_available_vis = inherited_effective_vis.nearest_available(level).unwrap();
                    let calculated_effective_vis = match current_vis {
                        Visibility::Public => nearest_available_vis,
                        Visibility::Restricted(_) => {
                            if current_vis.is_at_least(nearest_available_vis, &*self.r) {nearest_available_vis} else {current_vis}
                        }
                    };
                    current_effective_vis.update(calculated_effective_vis, level, &*self.r);
                }
            }
            if current_effective_vis_copy != current_effective_vis {
                self.changed = true;
                self.r.access_levels.set_effective_vis(current_id, current_effective_vis);
            }
        }
    }

    fn set_access_level_def_id(
        &mut self,
        def_id: LocalDefId,
        access_level: Option<AccessLevel>,
    ) -> Option<AccessLevel> {
        let old_level = self.r.access_levels.get_access_level(def_id);
        if old_level < access_level {
            self.r.access_levels.set_access_level(def_id, access_level.unwrap());
            self.changed = true;
            access_level
        } else {
            old_level
        }
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
                self.update_effective_vis(def_id, Visibility::Public, parent_id, AccessLevel::Public);
            }

            // Only exported `macro_rules!` items are public, but they always are
            ast::ItemKind::MacroDef(ref macro_def) if macro_def.macro_rules => {
                let parent_id = self.r.local_parent(def_id);
                let vis = self.r.visibilities.get(&def_id).unwrap().clone();
                self.update_effective_vis(def_id, vis, parent_id, AccessLevel::Public);
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
                        let vis = self.r.visibilities.get(&field_def_id).unwrap().clone();
                        self.update_effective_vis(
                            field_def_id,
                            vis,
                            variant_def_id,
                            AccessLevel::Public,
                        );
                    }
                }
            }

            ast::ItemKind::Struct(ref def, _) | ast::ItemKind::Union(ref def, _) => {
                for field in def.fields() {
                    let field_def_id = self.r.local_def_id(field.id);
                    let vis = self.r.visibilities.get(&field_def_id).unwrap();
                    self.update_effective_vis(field_def_id, *vis, def_id, AccessLevel::Public);
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
