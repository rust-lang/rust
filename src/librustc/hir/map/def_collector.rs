// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hir::map::definitions::*;

use hir;
use hir::intravisit;
use hir::def_id::{CRATE_DEF_INDEX, DefId, DefIndex};

use middle::cstore::InlinedItem;

use syntax::ast::*;
use syntax::ext::hygiene::Mark;
use syntax::visit;
use syntax::symbol::{Symbol, keywords};

/// Creates def ids for nodes in the HIR.
pub struct DefCollector<'a> {
    // If we are walking HIR (c.f., AST), we need to keep a reference to the
    // crate.
    hir_crate: Option<&'a hir::Crate>,
    definitions: &'a mut Definitions,
    parent_def: Option<DefIndex>,
    pub visit_macro_invoc: Option<&'a mut FnMut(MacroInvocationData)>,
}

pub struct MacroInvocationData {
    pub mark: Mark,
    pub def_index: DefIndex,
    pub const_integer: bool,
}

impl<'a> DefCollector<'a> {
    pub fn new(definitions: &'a mut Definitions) -> Self {
        DefCollector {
            hir_crate: None,
            definitions: definitions,
            parent_def: None,
            visit_macro_invoc: None,
        }
    }

    pub fn extend(parent_node: NodeId,
                  parent_def_path: DefPath,
                  parent_def_id: DefId,
                  definitions: &'a mut Definitions)
                  -> Self {
        let mut collector = DefCollector::new(definitions);

        assert_eq!(parent_def_path.krate, parent_def_id.krate);
        let root_path = Box::new(InlinedRootPath {
            data: parent_def_path.data,
            def_id: parent_def_id,
        });

        let def = collector.create_def(parent_node, DefPathData::InlinedRoot(root_path));
        collector.parent_def = Some(def);

        collector
    }

    pub fn collect_root(&mut self) {
        let root = self.create_def_with_parent(None, CRATE_NODE_ID, DefPathData::CrateRoot);
        assert_eq!(root, CRATE_DEF_INDEX);
        self.parent_def = Some(root);

        self.create_def_with_parent(Some(CRATE_DEF_INDEX), DUMMY_NODE_ID, DefPathData::Misc);
    }

    pub fn walk_item(&mut self, ii: &'a InlinedItem, krate: &'a hir::Crate) {
        self.hir_crate = Some(krate);
        ii.visit(self);
    }

    fn create_def(&mut self, node_id: NodeId, data: DefPathData) -> DefIndex {
        let parent_def = self.parent_def;
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

    pub fn with_parent<F: FnOnce(&mut Self)>(&mut self, parent_def: DefIndex, f: F) {
        let parent = self.parent_def;
        self.parent_def = Some(parent_def);
        f(self);
        self.parent_def = parent;
    }

    pub fn visit_ast_const_integer(&mut self, expr: &Expr) {
        match expr.node {
            // Find the node which will be used after lowering.
            ExprKind::Paren(ref inner) => return self.visit_ast_const_integer(inner),
            ExprKind::Mac(..) => return self.visit_macro_invoc(expr.id, true),
            // FIXME(eddyb) Closures should have separate
            // function definition IDs and expression IDs.
            ExprKind::Closure(..) => return,
            _ => {}
        }

        self.create_def(expr.id, DefPathData::Initializer);
    }

    fn visit_hir_const_integer(&mut self, expr: &hir::Expr) {
        // FIXME(eddyb) Closures should have separate
        // function definition IDs and expression IDs.
        if let hir::ExprClosure(..) = expr.node {
            return;
        }

        self.create_def(expr.id, DefPathData::Initializer);
    }

    fn visit_macro_invoc(&mut self, id: NodeId, const_integer: bool) {
        if let Some(ref mut visit) = self.visit_macro_invoc {
            visit(MacroInvocationData {
                mark: Mark::from_placeholder_id(id),
                const_integer: const_integer,
                def_index: self.parent_def.unwrap(),
            })
        }
    }
}

impl<'a> visit::Visitor for DefCollector<'a> {
    fn visit_item(&mut self, i: &Item) {
        debug!("visit_item: {:?}", i);

        // Pick the def data. This need not be unique, but the more
        // information we encapsulate into
        let def_data = match i.node {
            ItemKind::DefaultImpl(..) | ItemKind::Impl(..) =>
                DefPathData::Impl,
            ItemKind::Enum(..) | ItemKind::Struct(..) | ItemKind::Union(..) | ItemKind::Trait(..) |
            ItemKind::ExternCrate(..) | ItemKind::ForeignMod(..) | ItemKind::Ty(..) =>
                DefPathData::TypeNs(i.ident.name.as_str()),
            ItemKind::Mod(..) if i.ident == keywords::Invalid.ident() => {
                return visit::walk_item(self, i);
            }
            ItemKind::Mod(..) => DefPathData::Module(i.ident.name.as_str()),
            ItemKind::Static(..) | ItemKind::Const(..) | ItemKind::Fn(..) =>
                DefPathData::ValueNs(i.ident.name.as_str()),
            ItemKind::Mac(..) if i.id == DUMMY_NODE_ID => return, // Scope placeholder
            ItemKind::Mac(..) => return self.visit_macro_invoc(i.id, false),
            ItemKind::Use(..) => DefPathData::Misc,
        };
        let def = self.create_def(i.id, def_data);

        self.with_parent(def, |this| {
            match i.node {
                ItemKind::Enum(ref enum_definition, _) => {
                    for v in &enum_definition.variants {
                        let variant_def_index =
                            this.create_def(v.node.data.id(),
                                            DefPathData::EnumVariant(v.node.name.name.as_str()));
                        this.with_parent(variant_def_index, |this| {
                            for (index, field) in v.node.data.fields().iter().enumerate() {
                                let name = field.ident.map(|ident| ident.name)
                                    .unwrap_or_else(|| Symbol::intern(&index.to_string()));
                                this.create_def(field.id, DefPathData::Field(name.as_str()));
                            }

                            if let Some(ref expr) = v.node.disr_expr {
                                this.visit_ast_const_integer(expr);
                            }
                        });
                    }
                }
                ItemKind::Struct(ref struct_def, _) | ItemKind::Union(ref struct_def, _) => {
                    // If this is a tuple-like struct, register the constructor.
                    if !struct_def.is_struct() {
                        this.create_def(struct_def.id(),
                                        DefPathData::StructCtor);
                    }

                    for (index, field) in struct_def.fields().iter().enumerate() {
                        let name = field.ident.map(|ident| ident.name.as_str())
                            .unwrap_or(Symbol::intern(&index.to_string()).as_str());
                        this.create_def(field.id, DefPathData::Field(name));
                    }
                }
                _ => {}
            }
            visit::walk_item(this, i);
        });
    }

    fn visit_foreign_item(&mut self, foreign_item: &ForeignItem) {
        let def = self.create_def(foreign_item.id,
                                  DefPathData::ValueNs(foreign_item.ident.name.as_str()));

        self.with_parent(def, |this| {
            visit::walk_foreign_item(this, foreign_item);
        });
    }

    fn visit_generics(&mut self, generics: &Generics) {
        for ty_param in generics.ty_params.iter() {
            self.create_def(ty_param.id, DefPathData::TypeParam(ty_param.ident.name.as_str()));
        }

        visit::walk_generics(self, generics);
    }

    fn visit_trait_item(&mut self, ti: &TraitItem) {
        let def_data = match ti.node {
            TraitItemKind::Method(..) | TraitItemKind::Const(..) =>
                DefPathData::ValueNs(ti.ident.name.as_str()),
            TraitItemKind::Type(..) => DefPathData::TypeNs(ti.ident.name.as_str()),
            TraitItemKind::Macro(..) => return self.visit_macro_invoc(ti.id, false),
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
                DefPathData::ValueNs(ii.ident.name.as_str()),
            ImplItemKind::Type(..) => DefPathData::TypeNs(ii.ident.name.as_str()),
            ImplItemKind::Macro(..) => return self.visit_macro_invoc(ii.id, false),
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

        match pat.node {
            PatKind::Mac(..) => return self.visit_macro_invoc(pat.id, false),
            PatKind::Ident(_, id, _) => {
                let def = self.create_def(pat.id, DefPathData::Binding(id.node.name.as_str()));
                self.parent_def = Some(def);
            }
            _ => {}
        }

        visit::walk_pat(self, pat);
        self.parent_def = parent_def;
    }

    fn visit_expr(&mut self, expr: &Expr) {
        let parent_def = self.parent_def;

        match expr.node {
            ExprKind::Mac(..) => return self.visit_macro_invoc(expr.id, false),
            ExprKind::Repeat(_, ref count) => self.visit_ast_const_integer(count),
            ExprKind::Closure(..) => {
                let def = self.create_def(expr.id, DefPathData::ClosureExpr);
                self.parent_def = Some(def);
            }
            _ => {}
        }

        visit::walk_expr(self, expr);
        self.parent_def = parent_def;
    }

    fn visit_ty(&mut self, ty: &Ty) {
        match ty.node {
            TyKind::Mac(..) => return self.visit_macro_invoc(ty.id, false),
            TyKind::Array(_, ref length) => self.visit_ast_const_integer(length),
            TyKind::ImplTrait(..) => {
                self.create_def(ty.id, DefPathData::ImplTrait);
            }
            _ => {}
        }
        visit::walk_ty(self, ty);
    }

    fn visit_lifetime_def(&mut self, def: &LifetimeDef) {
        self.create_def(def.lifetime.id, DefPathData::LifetimeDef(def.lifetime.name.as_str()));
    }

    fn visit_macro_def(&mut self, macro_def: &MacroDef) {
        self.create_def(macro_def.id, DefPathData::MacroDef(macro_def.ident.name.as_str()));
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match stmt.node {
            StmtKind::Mac(..) => self.visit_macro_invoc(stmt.id, false),
            _ => visit::walk_stmt(self, stmt),
        }
    }
}

// We walk the HIR rather than the AST when reading items from metadata.
impl<'ast> intravisit::Visitor<'ast> for DefCollector<'ast> {
    fn visit_item(&mut self, i: &'ast hir::Item) {
        debug!("visit_item: {:?}", i);

        // Pick the def data. This need not be unique, but the more
        // information we encapsulate into
        let def_data = match i.node {
            hir::ItemDefaultImpl(..) | hir::ItemImpl(..) =>
                DefPathData::Impl,
            hir::ItemEnum(..) | hir::ItemStruct(..) | hir::ItemUnion(..) |
            hir::ItemTrait(..) | hir::ItemExternCrate(..) | hir::ItemMod(..) |
            hir::ItemForeignMod(..) | hir::ItemTy(..) =>
                DefPathData::TypeNs(i.name.as_str()),
            hir::ItemStatic(..) | hir::ItemConst(..) | hir::ItemFn(..) =>
                DefPathData::ValueNs(i.name.as_str()),
            hir::ItemUse(..) => DefPathData::Misc,
        };
        let def = self.create_def(i.id, def_data);

        self.with_parent(def, |this| {
            match i.node {
                hir::ItemEnum(ref enum_definition, _) => {
                    for v in &enum_definition.variants {
                        let variant_def_index =
                            this.create_def(v.node.data.id(),
                                            DefPathData::EnumVariant(v.node.name.as_str()));

                        this.with_parent(variant_def_index, |this| {
                            for field in v.node.data.fields() {
                                this.create_def(field.id,
                                                DefPathData::Field(field.name.as_str()));
                            }
                            if let Some(ref expr) = v.node.disr_expr {
                                this.visit_hir_const_integer(expr);
                            }
                        });
                    }
                }
                hir::ItemStruct(ref struct_def, _) |
                hir::ItemUnion(ref struct_def, _) => {
                    // If this is a tuple-like struct, register the constructor.
                    if !struct_def.is_struct() {
                        this.create_def(struct_def.id(),
                                        DefPathData::StructCtor);
                    }

                    for field in struct_def.fields() {
                        this.create_def(field.id, DefPathData::Field(field.name.as_str()));
                    }
                }
                _ => {}
            }
            intravisit::walk_item(this, i);
        });
    }

    fn visit_foreign_item(&mut self, foreign_item: &'ast hir::ForeignItem) {
        let def = self.create_def(foreign_item.id,
                                  DefPathData::ValueNs(foreign_item.name.as_str()));

        self.with_parent(def, |this| {
            intravisit::walk_foreign_item(this, foreign_item);
        });
    }

    fn visit_generics(&mut self, generics: &'ast hir::Generics) {
        for ty_param in generics.ty_params.iter() {
            self.create_def(ty_param.id, DefPathData::TypeParam(ty_param.name.as_str()));
        }

        intravisit::walk_generics(self, generics);
    }

    fn visit_trait_item(&mut self, ti: &'ast hir::TraitItem) {
        let def_data = match ti.node {
            hir::MethodTraitItem(..) | hir::ConstTraitItem(..) =>
                DefPathData::ValueNs(ti.name.as_str()),
            hir::TypeTraitItem(..) => DefPathData::TypeNs(ti.name.as_str()),
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
                DefPathData::ValueNs(ii.name.as_str()),
            hir::ImplItemKind::Type(..) => DefPathData::TypeNs(ii.name.as_str()),
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
            let def = self.create_def(pat.id, DefPathData::Binding(name.node.as_str()));
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
        if let hir::TyArray(_, ref length) = ty.node {
            self.visit_hir_const_integer(length);
        }
        if let hir::TyImplTrait(..) = ty.node {
            self.create_def(ty.id, DefPathData::ImplTrait);
        }
        intravisit::walk_ty(self, ty);
    }

    fn visit_lifetime_def(&mut self, def: &'ast hir::LifetimeDef) {
        self.create_def(def.lifetime.id, DefPathData::LifetimeDef(def.lifetime.name.as_str()));
    }

    fn visit_macro_def(&mut self, macro_def: &'ast hir::MacroDef) {
        self.create_def(macro_def.id, DefPathData::MacroDef(macro_def.name.as_str()));
    }
}
