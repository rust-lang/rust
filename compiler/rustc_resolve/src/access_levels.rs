use rustc_ast::ast;
use rustc_ast::visit;
use rustc_ast::visit::Visitor;
use rustc_ast::Crate;
use rustc_ast::EnumDef;
use rustc_ast::ForeignMod;
use rustc_ast::NodeId;
use rustc_ast_lowering::ResolverAstLowering;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::def_id::CRATE_DEF_ID;
use rustc_middle::middle::privacy::AccessLevel;
use rustc_middle::ty::Visibility;
use rustc_span::sym;

use crate::imports::ImportKind;
use crate::BindingKey;
use crate::NameBinding;
use crate::NameBindingKind;
use crate::Resolver;

pub struct AccessLevelsVisitor<'r, 'a> {
    r: &'r mut Resolver<'a>,
    prev_level: Option<AccessLevel>,
    changed: bool,
}

impl<'r, 'a> AccessLevelsVisitor<'r, 'a> {
    /// Fills the `Resolver::access_levels` table with public & exported items
    /// For now, this doesn't resolve macros (FIXME) and cannot resolve Impl, as we
    /// need access to a TyCtxt for that.
    pub fn compute_access_levels<'c>(r: &'r mut Resolver<'a>, krate: &'c Crate) {
        let mut visitor =
            AccessLevelsVisitor { r, changed: false, prev_level: Some(AccessLevel::Public) };

        visitor.set_access_level_def_id(CRATE_DEF_ID, Some(AccessLevel::Public));
        visitor.set_exports_access_level(CRATE_DEF_ID);

        while visitor.changed {
            visitor.reset();
            visit::walk_crate(&mut visitor, krate);
        }

        tracing::info!("resolve::access_levels: {:#?}", r.access_levels);
    }

    fn reset(&mut self) {
        self.changed = false;
        self.prev_level = Some(AccessLevel::Public);
    }

    /// Update the access level of the exports of the given module accordingly. The module access
    /// level has to be Exported or Public.
    /// This will also follow `use` chains (see PrivacyVisitor::set_import_binding_access_level).
    fn set_exports_access_level(&mut self, module_id: LocalDefId) {
        assert!(self.r.module_map.contains_key(&&module_id.to_def_id()));

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

        let module_level = self.r.access_levels.map.get(&module_id).copied();
        assert!(module_level >= Some(AccessLevel::Exported));

        if let Some(exports) = self.r.reexport_map.get(&module_id) {
            let pub_exports = exports
                .iter()
                .filter(|ex| ex.vis == Visibility::Public)
                .cloned()
                .collect::<Vec<_>>();

            let module = self.r.get_module(module_id.to_def_id()).unwrap();
            for export in pub_exports.into_iter() {
                if let Some(export_def_id) = export.res.opt_def_id().and_then(|id| id.as_local()) {
                    self.set_access_level_def_id(export_def_id, Some(AccessLevel::Exported));
                }

                if let Some(ns) = export.res.ns() {
                    let key = BindingKey { ident: export.ident, ns, disambiguator: 0 };
                    let name_res = self.r.resolution(module, key);
                    if let Some(binding) = name_res.borrow().binding() {
                        set_import_binding_access_level(self, binding, module_level)
                    }
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
        let inherited_item_level = match item.kind {
            // Resolved in rustc_privacy when types are available
            ast::ItemKind::Impl(..) => return,

            // Only exported `macro_rules!` items are public, but they always are
            ast::ItemKind::MacroDef(..) => {
                let is_macro_export =
                    item.attrs.iter().any(|attr| attr.has_name(sym::macro_export));
                if is_macro_export { Some(AccessLevel::Public) } else { None }
            }

            // Foreign modules inherit level from parents.
            ast::ItemKind::ForeignMod(..) => self.prev_level,

            // Other `pub` items inherit levels from parents.
            ast::ItemKind::ExternCrate(..)
            | ast::ItemKind::Use(..)
            | ast::ItemKind::Static(..)
            | ast::ItemKind::Const(..)
            | ast::ItemKind::Fn(..)
            | ast::ItemKind::Mod(..)
            | ast::ItemKind::GlobalAsm(..)
            | ast::ItemKind::TyAlias(..)
            | ast::ItemKind::Enum(..)
            | ast::ItemKind::Struct(..)
            | ast::ItemKind::Union(..)
            | ast::ItemKind::Trait(..)
            | ast::ItemKind::TraitAlias(..) => {
                if item.vis.kind.is_pub() {
                    self.prev_level
                } else {
                    None
                }
            }

            // Should be unreachable at this stage
            ast::ItemKind::MacCall(..) => panic!(
                "ast::ItemKind::MacCall encountered, this should not anymore appear at this stage"
            ),
        };

        let access_level = self.set_access_level(item.id, inherited_item_level);

        // Set access level of nested items.
        // If it's a mod, also make the visitor walk all of its items
        match item.kind {
            ast::ItemKind::Mod(..) => {
                if access_level.is_some() {
                    self.set_exports_access_level(self.r.local_def_id(item.id));
                }

                let orig_level = std::mem::replace(&mut self.prev_level, access_level);
                visit::walk_item(self, item);
                self.prev_level = orig_level;
            }

            ast::ItemKind::ForeignMod(ForeignMod { ref items, .. }) => {
                for nested in items {
                    if nested.vis.kind.is_pub() {
                        self.set_access_level(nested.id, access_level);
                    }
                }
            }
            ast::ItemKind::Enum(EnumDef { ref variants }, _) => {
                for variant in variants {
                    let variant_level = self.set_access_level(variant.id, access_level);
                    if let Some(ctor_id) = variant.data.ctor_id() {
                        self.set_access_level(ctor_id, access_level);
                    }

                    for field in variant.data.fields() {
                        self.set_access_level(field.id, variant_level);
                    }
                }
            }
            ast::ItemKind::Struct(ref def, _) | ast::ItemKind::Union(ref def, _) => {
                if let Some(ctor_id) = def.ctor_id() {
                    self.set_access_level(ctor_id, access_level);
                }

                for field in def.fields() {
                    if field.vis.kind.is_pub() {
                        self.set_access_level(field.id, access_level);
                    }
                }
            }
            ast::ItemKind::Trait(ref trait_kind) => {
                for nested in trait_kind.items.iter() {
                    self.set_access_level(nested.id, access_level);
                }
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

            // Unreachable kinds
            ast::ItemKind::Impl(..) | ast::ItemKind::MacCall(..) => unreachable!(),
        }
    }
}
