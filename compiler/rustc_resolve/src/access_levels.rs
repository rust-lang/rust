use crate::imports::ImportKind;
use crate::NameBinding;
use crate::NameBindingKind;
use crate::Resolver;
use rustc_ast::ast;
use rustc_ast::visit;
use rustc_ast::visit::Visitor;
use rustc_ast::Crate;
use rustc_ast::EnumDef;
use rustc_ast::NodeId;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::def_id::CRATE_DEF_ID;
use rustc_middle::middle::privacy::AccessLevel;
use rustc_middle::ty::DefIdTree;
use rustc_span::sym;

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
        let module_level = self.r.access_levels.map.get(&module_id).copied();
        if !module_level.is_some() {
            return;
        }
        // Set the given binding access level to `AccessLevel::Public` and
        // sets the rest of the `use` chain to `AccessLevel::Exported` until
        // we hit the actual exported item.
        let set_import_binding_access_level =
            |this: &mut Self, mut binding: &NameBinding<'a>, mut access_level| {
                while let NameBindingKind::Import { binding: nested_binding, import, .. } =
                    binding.kind
                {
                    this.set_access_level(import.id, access_level);
                    if let ImportKind::Single { additional_ids, .. } = import.kind {
                        this.set_access_level(additional_ids.0, access_level);
                        this.set_access_level(additional_ids.1, access_level);
                    }

                    access_level = Some(AccessLevel::Exported);
                    binding = nested_binding;
                }
            };

        let module = self.r.get_module(module_id.to_def_id()).unwrap();
        let resolutions = self.r.resolutions(module);

        for (.., name_resolution) in resolutions.borrow().iter() {
            if let Some(binding) = name_resolution.borrow().binding() && binding.vis.is_public() && !binding.is_ambiguity() {
                let access_level = match binding.is_import() {
                    true => {
                        set_import_binding_access_level(self, binding, module_level);
                        Some(AccessLevel::Exported)
                    },
                    false => module_level,
                };
                if let Some(def_id) = binding.res().opt_def_id().and_then(|id| id.as_local()) {
                    self.set_access_level_def_id(def_id, access_level);
                }
            }
        }
    }

    /// Sets the access level of the `LocalDefId` corresponding to the given `NodeId`.
    /// This function will panic if the `NodeId` does not have a `LocalDefId`
    fn set_access_level(
        &mut self,
        node_id: NodeId,
        access_level: Option<AccessLevel>,
    ) -> Option<AccessLevel> {
        self.set_access_level_def_id(self.r.local_def_id(node_id), access_level)
    }

    fn set_access_level_def_id(
        &mut self,
        def_id: LocalDefId,
        access_level: Option<AccessLevel>,
    ) -> Option<AccessLevel> {
        let old_level = self.r.access_levels.map.get(&def_id).copied();
        if old_level < access_level {
            self.r.access_levels.map.insert(def_id, access_level.unwrap());
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
                let parent_level =
                    self.r.access_levels.map.get(&self.r.local_parent(def_id)).copied();
                self.set_access_level(item.id, parent_level);
            }

            // Only exported `macro_rules!` items are public, but they always are
            ast::ItemKind::MacroDef(ref macro_def) if macro_def.macro_rules => {
                if item.attrs.iter().any(|attr| attr.has_name(sym::macro_export)) {
                    self.set_access_level(item.id, Some(AccessLevel::Public));
                }
            }

            ast::ItemKind::Mod(..) => {
                self.set_bindings_access_level(def_id);
                visit::walk_item(self, item);
            }

            ast::ItemKind::Enum(EnumDef { ref variants }, _) => {
                self.set_bindings_access_level(def_id);
                for variant in variants {
                    let variant_def_id = self.r.local_def_id(variant.id);
                    let variant_level = self.r.access_levels.map.get(&variant_def_id).copied();
                    for field in variant.data.fields() {
                        self.set_access_level(field.id, variant_level);
                    }
                }
            }

            ast::ItemKind::Struct(ref def, _) | ast::ItemKind::Union(ref def, _) => {
                let inherited_level = self.r.access_levels.map.get(&def_id).copied();
                for field in def.fields() {
                    if field.vis.kind.is_pub() {
                        self.set_access_level(field.id, inherited_level);
                    }
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
