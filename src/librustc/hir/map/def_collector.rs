// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::*;

use hir;
use hir::intravisit;
use hir::def_id::{CRATE_DEF_INDEX, DefId, DefIndex};

use middle::cstore::InlinedItem;

use syntax::ast::*;
use syntax::visit;
use syntax::parse::token;

/// Creates def ids for nodes in the HIR.
pub struct DefCollector<'ast> {
    // If we are walking HIR (c.f., AST), we need to keep a reference to the
    // crate.
    hir_crate: Option<&'ast hir::Crate>,
    pub definitions: Definitions,
    parent_def: Option<DefIndex>,
}

impl<'ast> DefCollector<'ast> {
    pub fn root() -> DefCollector<'ast> {
        let mut collector = DefCollector {
            hir_crate: None,
            definitions: Definitions::new(),
            parent_def: None,
        };
        let root = collector.create_def_with_parent(None, CRATE_NODE_ID, DefPathData::CrateRoot);
        assert_eq!(root, CRATE_DEF_INDEX);
        collector.parent_def = Some(root);

        collector.create_def_with_parent(Some(CRATE_DEF_INDEX), DUMMY_NODE_ID, DefPathData::Misc);

        collector
    }

    pub fn extend(parent_node: NodeId,
                  parent_def_path: DefPath,
                  parent_def_id: DefId,
                  definitions: Definitions)
                  -> DefCollector<'ast> {
        let mut collector = DefCollector {
            hir_crate: None,
            parent_def: None,
            definitions: definitions,
        };

        assert_eq!(parent_def_path.krate, parent_def_id.krate);
        let root_path = Box::new(InlinedRootPath {
            data: parent_def_path.data,
            def_id: parent_def_id,
        });

        let def = collector.create_def(parent_node, DefPathData::InlinedRoot(root_path));
        collector.parent_def = Some(def);

        collector
    }

    pub fn walk_item(&mut self, ii: &'ast InlinedItem, krate: &'ast hir::Crate) {
        self.hir_crate = Some(krate);
        ii.visit(self);
    }

    fn parent_def(&self) -> Option<DefIndex> {
        self.parent_def
    }

    fn create_def(&mut self, node_id: NodeId, data: DefPathData) -> DefIndex {
        let parent_def = self.parent_def();
        debug!("create_def(node_id={:?}, data={:?}, parent_def={:?})", node_id, data, parent_def);
        self.definitions.create_def_with_parent(parent_def, node_id, data)
    }

    fn create_def_with_parent(&mut self,
                              parent: Option<DefIndex>,
                              node_id: NodeId,
                              data: DefPathData)
                              -> DefIndex {
        self.definitions.create_def_with_parent(parent, node_id, data)
    }

    fn with_parent<F: FnOnce(&mut Self)>(&mut self, parent_def: DefIndex, f: F) {
        let parent = self.parent_def;
        self.parent_def = Some(parent_def);
        f(self);
        self.parent_def = parent;
    }

    fn visit_ast_const_integer(&mut self, expr: &Expr) {
        // Find the node which will be used after lowering.
        if let ExprKind::Paren(ref inner) = expr.node {
            return self.visit_ast_const_integer(inner);
        }

        // FIXME(eddyb) Closures should have separate
        // function definition IDs and expression IDs.
        if let ExprKind::Closure(..) = expr.node {
            return;
        }

        self.create_def(expr.id, DefPathData::Initializer);
    }

    fn visit_hir_const_integer(&mut self, expr: &'ast hir::Expr) {
        // FIXME(eddyb) Closures should have separate
        // function definition IDs and expression IDs.
        if let hir::ExprClosure(..) = expr.node {
            return;
        }

        self.create_def(expr.id, DefPathData::Initializer);
    }
}

impl<'ast> visit::Visitor for DefCollector<'ast> {
    fn visit_item(&mut self, i: &Item) {
        debug!("visit_item: {:?}", i);

        // Pick the def data. This need not be unique, but the more
        // information we encapsulate into
        let def_data = match i.node {
            ItemKind::DefaultImpl(..) | ItemKind::Impl(..) =>
                DefPathData::Impl,
            ItemKind::Enum(..) | ItemKind::Struct(..) | ItemKind::Trait(..) |
            ItemKind::ExternCrate(..) | ItemKind::ForeignMod(..) | ItemKind::Ty(..) =>
                DefPathData::TypeNs(i.ident.name),
            ItemKind::Mod(..) => DefPathData::Module(i.ident.name),
            ItemKind::Static(..) | ItemKind::Const(..) | ItemKind::Fn(..) =>
                DefPathData::ValueNs(i.ident.name),
            ItemKind::Mac(..) => DefPathData::MacroDef(i.ident.name),
            ItemKind::Use(..) => DefPathData::Misc,
        };
        let def = self.create_def(i.id, def_data);

        self.with_parent(def, |this| {
            match i.node {
                ItemKind::Enum(ref enum_definition, _) => {
                    for v in &enum_definition.variants {
                        let variant_def_index =
                            this.create_def(v.node.data.id(),
                                            DefPathData::EnumVariant(v.node.name.name));
                        this.with_parent(variant_def_index, |this| {
                            for (index, field) in v.node.data.fields().iter().enumerate() {
                                let name = field.ident.map(|ident| ident.name)
                                    .unwrap_or_else(|| token::intern(&index.to_string()));
                                this.create_def(field.id, DefPathData::Field(name));
                            }

                            if let Some(ref expr) = v.node.disr_expr {
                                this.visit_ast_const_integer(expr);
                            }
                        });
                    }
                }
                ItemKind::Struct(ref struct_def, _) => {
                    // If this is a tuple-like struct, register the constructor.
                    if !struct_def.is_struct() {
                        this.create_def(struct_def.id(),
                                        DefPathData::StructCtor);
                    }

                    for (index, field) in struct_def.fields().iter().enumerate() {
                        let name = field.ident.map(|ident| ident.name)
                            .unwrap_or(token::intern(&index.to_string()));
                        this.create_def(field.id, DefPathData::Field(name));
                    }
                }
                _ => {}
            }
            visit::walk_item(this, i);
        });
    }

    fn visit_foreign_item(&mut self, foreign_item: &ForeignItem) {
        let def = self.create_def(foreign_item.id, DefPathData::ValueNs(foreign_item.ident.name));

        self.with_parent(def, |this| {
            visit::walk_foreign_item(this, foreign_item);
        });
    }

    fn visit_generics(&mut self, generics: &Generics) {
        for ty_param in generics.ty_params.iter() {
            self.create_def(ty_param.id, DefPathData::TypeParam(ty_param.ident.name));
        }

        visit::walk_generics(self, generics);
    }

    fn visit_trait_item(&mut self, ti: &TraitItem) {
        let def_data = match ti.node {
            TraitItemKind::Method(..) | TraitItemKind::Const(..) =>
                DefPathData::ValueNs(ti.ident.name),
            TraitItemKind::Type(..) => DefPathData::TypeNs(ti.ident.name),
            TraitItemKind::Macro(..) => DefPathData::MacroDef(ti.ident.name),
        };

        let def = self.create_def(ti.id, def_data);
        self.with_parent(def, |this| {
            if let TraitItemKind::Const(_, Some(ref expr)) = ti.node {
                this.create_def(expr.id, DefPathData::Initializer);
            }

            visit::walk_trait_item(this, ti);
        });
    }

    fn visit_impl_item(&mut self, ii: &ImplItem) {
        let def_data = match ii.node {
            ImplItemKind::Method(..) | ImplItemKind::Const(..) =>
                DefPathData::ValueNs(ii.ident.name),
            ImplItemKind::Type(..) => DefPathData::TypeNs(ii.ident.name),
            ImplItemKind::Macro(..) => DefPathData::MacroDef(ii.ident.name),
        };

        let def = self.create_def(ii.id, def_data);
        self.with_parent(def, |this| {
            if let ImplItemKind::Const(_, ref expr) = ii.node {
                this.create_def(expr.id, DefPathData::Initializer);
            }

            visit::walk_impl_item(this, ii);
        });
    }

    fn visit_pat(&mut self, pat: &Pat) {
        let parent_def = self.parent_def;

        if let PatKind::Ident(_, id, _) = pat.node {
            let def = self.create_def(pat.id, DefPathData::Binding(id.node.name));
            self.parent_def = Some(def);
        }

        visit::walk_pat(self, pat);
        self.parent_def = parent_def;
    }

    fn visit_expr(&mut self, expr: &Expr) {
        let parent_def = self.parent_def;

        if let ExprKind::Repeat(_, ref count) = expr.node {
            self.visit_ast_const_integer(count);
        }

        if let ExprKind::Closure(..) = expr.node {
            let def = self.create_def(expr.id, DefPathData::ClosureExpr);
            self.parent_def = Some(def);
        }

        visit::walk_expr(self, expr);
        self.parent_def = parent_def;
    }

    fn visit_ty(&mut self, ty: &Ty) {
        if let TyKind::FixedLengthVec(_, ref length) = ty.node {
            self.visit_ast_const_integer(length);
        }
        visit::walk_ty(self, ty);
    }

    fn visit_lifetime_def(&mut self, def: &LifetimeDef) {
        self.create_def(def.lifetime.id, DefPathData::LifetimeDef(def.lifetime.name));
    }

    fn visit_macro_def(&mut self, macro_def: &MacroDef) {
        self.create_def(macro_def.id, DefPathData::MacroDef(macro_def.ident.name));
    }
}

// We walk the HIR rather than the AST when reading items from metadata.
impl<'ast> intravisit::Visitor<'ast> for DefCollector<'ast> {
    /// Because we want to track parent items and so forth, enable
    /// deep walking so that we walk nested items in the context of
    /// their outer items.
    fn visit_nested_item(&mut self, item_id: hir::ItemId) {
        debug!("visit_nested_item: {:?}", item_id);
        let item = self.hir_crate.unwrap().item(item_id.id);
        self.visit_item(item)
    }

    fn visit_item(&mut self, i: &'ast hir::Item) {
        debug!("visit_item: {:?}", i);

        // Pick the def data. This need not be unique, but the more
        // information we encapsulate into
        let def_data = match i.node {
            hir::ItemDefaultImpl(..) | hir::ItemImpl(..) =>
                DefPathData::Impl,
            hir::ItemEnum(..) | hir::ItemStruct(..) | hir::ItemTrait(..) |
            hir::ItemExternCrate(..) | hir::ItemMod(..) | hir::ItemForeignMod(..) |
            hir::ItemTy(..) =>
                DefPathData::TypeNs(i.name),
            hir::ItemStatic(..) | hir::ItemConst(..) | hir::ItemFn(..) =>
                DefPathData::ValueNs(i.name),
            hir::ItemUse(..) => DefPathData::Misc,
        };
        let def = self.create_def(i.id, def_data);

        self.with_parent(def, |this| {
            match i.node {
                hir::ItemEnum(ref enum_definition, _) => {
                    for v in &enum_definition.variants {
                        let variant_def_index =
                            this.create_def(v.node.data.id(),
                                            DefPathData::EnumVariant(v.node.name));

                        this.with_parent(variant_def_index, |this| {
                            for field in v.node.data.fields() {
                                this.create_def(field.id,
                                                DefPathData::Field(field.name));
                            }
                            if let Some(ref expr) = v.node.disr_expr {
                                this.visit_hir_const_integer(expr);
                            }
                        });
                    }
                }
                hir::ItemStruct(ref struct_def, _) => {
                    // If this is a tuple-like struct, register the constructor.
                    if !struct_def.is_struct() {
                        this.create_def(struct_def.id(),
                                        DefPathData::StructCtor);
                    }

                    for field in struct_def.fields() {
                        this.create_def(field.id, DefPathData::Field(field.name));
                    }
                }
                _ => {}
            }
            intravisit::walk_item(this, i);
        });
    }

    fn visit_foreign_item(&mut self, foreign_item: &'ast hir::ForeignItem) {
        let def = self.create_def(foreign_item.id, DefPathData::ValueNs(foreign_item.name));

        self.with_parent(def, |this| {
            intravisit::walk_foreign_item(this, foreign_item);
        });
    }

    fn visit_generics(&mut self, generics: &'ast hir::Generics) {
        for ty_param in generics.ty_params.iter() {
            self.create_def(ty_param.id, DefPathData::TypeParam(ty_param.name));
        }

        intravisit::walk_generics(self, generics);
    }

    fn visit_trait_item(&mut self, ti: &'ast hir::TraitItem) {
        let def_data = match ti.node {
            hir::MethodTraitItem(..) | hir::ConstTraitItem(..) =>
                DefPathData::ValueNs(ti.name),
            hir::TypeTraitItem(..) => DefPathData::TypeNs(ti.name),
        };

        let def = self.create_def(ti.id, def_data);
        self.with_parent(def, |this| {
            if let hir::ConstTraitItem(_, Some(ref expr)) = ti.node {
                this.create_def(expr.id, DefPathData::Initializer);
            }

            intravisit::walk_trait_item(this, ti);
        });
    }

    fn visit_impl_item(&mut self, ii: &'ast hir::ImplItem) {
        let def_data = match ii.node {
            hir::ImplItemKind::Method(..) | hir::ImplItemKind::Const(..) =>
                DefPathData::ValueNs(ii.name),
            hir::ImplItemKind::Type(..) => DefPathData::TypeNs(ii.name),
        };

        let def = self.create_def(ii.id, def_data);
        self.with_parent(def, |this| {
            if let hir::ImplItemKind::Const(_, ref expr) = ii.node {
                this.create_def(expr.id, DefPathData::Initializer);
            }

            intravisit::walk_impl_item(this, ii);
        });
    }

    fn visit_pat(&mut self, pat: &'ast hir::Pat) {
        let parent_def = self.parent_def;

        if let hir::PatKind::Binding(_, name, _) = pat.node {
            let def = self.create_def(pat.id, DefPathData::Binding(name.node));
            self.parent_def = Some(def);
        }

        intravisit::walk_pat(self, pat);
        self.parent_def = parent_def;
    }

    fn visit_expr(&mut self, expr: &'ast hir::Expr) {
        let parent_def = self.parent_def;

        if let hir::ExprRepeat(_, ref count) = expr.node {
            self.visit_hir_const_integer(count);
        }

        if let hir::ExprClosure(..) = expr.node {
            let def = self.create_def(expr.id, DefPathData::ClosureExpr);
            self.parent_def = Some(def);
        }

        intravisit::walk_expr(self, expr);
        self.parent_def = parent_def;
    }

    fn visit_ty(&mut self, ty: &'ast hir::Ty) {
        if let hir::TyFixedLengthVec(_, ref length) = ty.node {
            self.visit_hir_const_integer(length);
        }
        intravisit::walk_ty(self, ty);
    }

    fn visit_lifetime_def(&mut self, def: &'ast hir::LifetimeDef) {
        self.create_def(def.lifetime.id, DefPathData::LifetimeDef(def.lifetime.name));
    }

    fn visit_macro_def(&mut self, macro_def: &'ast hir::MacroDef) {
        self.create_def(macro_def.id, DefPathData::MacroDef(macro_def.name));
    }
}
