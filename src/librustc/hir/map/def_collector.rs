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
use hir::def_id::{CRATE_DEF_INDEX, DefIndex, DefIndexAddressSpace};
use session::CrateDisambiguator;

use syntax::ast::*;
use syntax::ext::hygiene::Mark;
use syntax::visit;
use syntax::symbol::keywords;
use syntax::symbol::Symbol;
use syntax::parse::token::{self, Token};

use hir::map::{ITEM_LIKE_SPACE, REGULAR_SPACE};

/// Creates def ids for nodes in the AST.
pub struct DefCollector<'a> {
    definitions: &'a mut Definitions,
    parent_def: Option<DefIndex>,
    expansion: Mark,
    pub visit_macro_invoc: Option<&'a mut FnMut(MacroInvocationData)>,
}

pub struct MacroInvocationData {
    pub mark: Mark,
    pub def_index: DefIndex,
    pub const_expr: bool,
}

impl<'a> DefCollector<'a> {
    pub fn new(definitions: &'a mut Definitions, expansion: Mark) -> Self {
        DefCollector {
            definitions,
            expansion,
            parent_def: None,
            visit_macro_invoc: None,
        }
    }

    pub fn collect_root(&mut self,
                        crate_name: &str,
                        crate_disambiguator: CrateDisambiguator) {
        let root = self.definitions.create_root_def(crate_name,
                                                    crate_disambiguator);
        assert_eq!(root, CRATE_DEF_INDEX);
        self.parent_def = Some(root);
    }

    fn create_def(&mut self,
                  node_id: NodeId,
                  data: DefPathData,
                  address_space: DefIndexAddressSpace)
                  -> DefIndex {
        let parent_def = self.parent_def.unwrap();
        debug!("create_def(node_id={:?}, data={:?}, parent_def={:?})", node_id, data, parent_def);
        self.definitions
            .create_def_with_parent(parent_def, node_id, data, address_space, self.expansion)
    }

    pub fn with_parent<F: FnOnce(&mut Self)>(&mut self, parent_def: DefIndex, f: F) {
        let parent = self.parent_def;
        self.parent_def = Some(parent_def);
        f(self);
        self.parent_def = parent;
    }

    pub fn visit_const_expr(&mut self, expr: &Expr) {
        match expr.node {
            // Find the node which will be used after lowering.
            ExprKind::Paren(ref inner) => return self.visit_const_expr(inner),
            ExprKind::Mac(..) => return self.visit_macro_invoc(expr.id, true),
            // FIXME(eddyb) Closures should have separate
            // function definition IDs and expression IDs.
            ExprKind::Closure(..) => return,
            _ => {}
        }

        self.create_def(expr.id, DefPathData::Initializer, REGULAR_SPACE);
    }

    fn visit_macro_invoc(&mut self, id: NodeId, const_expr: bool) {
        if let Some(ref mut visit) = self.visit_macro_invoc {
            visit(MacroInvocationData {
                mark: id.placeholder_to_mark(),
                const_expr,
                def_index: self.parent_def.unwrap(),
            })
        }
    }
}

impl<'a> visit::Visitor<'a> for DefCollector<'a> {
    fn visit_item(&mut self, i: &'a Item) {
        debug!("visit_item: {:?}", i);

        // Pick the def data. This need not be unique, but the more
        // information we encapsulate into
        let def_data = match i.node {
            ItemKind::Impl(..) => DefPathData::Impl,
            ItemKind::Enum(..) | ItemKind::Struct(..) | ItemKind::Union(..) |
            ItemKind::Trait(..) | ItemKind::TraitAlias(..) |
            ItemKind::ExternCrate(..) | ItemKind::ForeignMod(..) | ItemKind::Ty(..) =>
                DefPathData::TypeNs(i.ident.name.as_str()),
            ItemKind::Mod(..) if i.ident == keywords::Invalid.ident() => {
                return visit::walk_item(self, i);
            }
            ItemKind::Mod(..) => DefPathData::Module(i.ident.name.as_str()),
            ItemKind::Static(..) | ItemKind::Const(..) | ItemKind::Fn(..) =>
                DefPathData::ValueNs(i.ident.name.as_str()),
            ItemKind::MacroDef(..) => DefPathData::MacroDef(i.ident.name.as_str()),
            ItemKind::Mac(..) => return self.visit_macro_invoc(i.id, false),
            ItemKind::GlobalAsm(..) => DefPathData::Misc,
            ItemKind::Use(..) => {
                return visit::walk_item(self, i);
            }
        };
        let def = self.create_def(i.id, def_data, ITEM_LIKE_SPACE);

        self.with_parent(def, |this| {
            match i.node {
                ItemKind::Enum(ref enum_definition, _) => {
                    for v in &enum_definition.variants {
                        let variant_def_index =
                            this.create_def(v.node.data.id(),
                                            DefPathData::EnumVariant(v.node.name.name.as_str()),
                                            REGULAR_SPACE);
                        this.with_parent(variant_def_index, |this| {
                            for (index, field) in v.node.data.fields().iter().enumerate() {
                                let name = field.ident.map(|ident| ident.name)
                                    .unwrap_or_else(|| Symbol::intern(&index.to_string()));
                                this.create_def(field.id,
                                                DefPathData::Field(name.as_str()),
                                                REGULAR_SPACE);
                            }

                            if let Some(ref expr) = v.node.disr_expr {
                                this.visit_const_expr(expr);
                            }
                        });
                    }
                }
                ItemKind::Struct(ref struct_def, _) | ItemKind::Union(ref struct_def, _) => {
                    // If this is a tuple-like struct, register the constructor.
                    if !struct_def.is_struct() {
                        this.create_def(struct_def.id(),
                                        DefPathData::StructCtor,
                                        REGULAR_SPACE);
                    }

                    for (index, field) in struct_def.fields().iter().enumerate() {
                        let name = field.ident.map(|ident| ident.name)
                            .unwrap_or_else(|| Symbol::intern(&index.to_string()));
                        this.create_def(field.id, DefPathData::Field(name.as_str()), REGULAR_SPACE);
                    }
                }
                _ => {}
            }
            visit::walk_item(this, i);
        });
    }

    fn visit_use_tree(&mut self, use_tree: &'a UseTree, id: NodeId, _nested: bool) {
        self.create_def(id, DefPathData::Misc, ITEM_LIKE_SPACE);
        visit::walk_use_tree(self, use_tree, id);
    }

    fn visit_foreign_item(&mut self, foreign_item: &'a ForeignItem) {
        let def = self.create_def(foreign_item.id,
                                  DefPathData::ValueNs(foreign_item.ident.name.as_str()),
                                  REGULAR_SPACE);

        self.with_parent(def, |this| {
            visit::walk_foreign_item(this, foreign_item);
        });
    }

    fn visit_generic_param(&mut self, param: &'a GenericParam) {
        match *param {
            GenericParam::Lifetime(ref lifetime_def) => {
                self.create_def(
                    lifetime_def.lifetime.id,
                    DefPathData::LifetimeDef(lifetime_def.lifetime.ident.name.as_str()),
                    REGULAR_SPACE
                );
            }
            GenericParam::Type(ref ty_param) => {
                self.create_def(
                    ty_param.id,
                    DefPathData::TypeParam(ty_param.ident.name.as_str()),
                    REGULAR_SPACE
                );
            }
        }

        visit::walk_generic_param(self, param);
    }

    fn visit_trait_item(&mut self, ti: &'a TraitItem) {
        let def_data = match ti.node {
            TraitItemKind::Method(..) | TraitItemKind::Const(..) =>
                DefPathData::ValueNs(ti.ident.name.as_str()),
            TraitItemKind::Type(..) => DefPathData::TypeNs(ti.ident.name.as_str()),
            TraitItemKind::Macro(..) => return self.visit_macro_invoc(ti.id, false),
        };

        let def = self.create_def(ti.id, def_data, ITEM_LIKE_SPACE);
        self.with_parent(def, |this| {
            if let TraitItemKind::Const(_, Some(ref expr)) = ti.node {
                this.visit_const_expr(expr);
            }

            visit::walk_trait_item(this, ti);
        });
    }

    fn visit_impl_item(&mut self, ii: &'a ImplItem) {
        let def_data = match ii.node {
            ImplItemKind::Method(..) | ImplItemKind::Const(..) =>
                DefPathData::ValueNs(ii.ident.name.as_str()),
            ImplItemKind::Type(..) => DefPathData::TypeNs(ii.ident.name.as_str()),
            ImplItemKind::Macro(..) => return self.visit_macro_invoc(ii.id, false),
        };

        let def = self.create_def(ii.id, def_data, ITEM_LIKE_SPACE);
        self.with_parent(def, |this| {
            if let ImplItemKind::Const(_, ref expr) = ii.node {
                this.visit_const_expr(expr);
            }

            visit::walk_impl_item(this, ii);
        });
    }

    fn visit_pat(&mut self, pat: &'a Pat) {
        match pat.node {
            PatKind::Mac(..) => return self.visit_macro_invoc(pat.id, false),
            _ => visit::walk_pat(self, pat),
        }
    }

    fn visit_expr(&mut self, expr: &'a Expr) {
        let parent_def = self.parent_def;

        match expr.node {
            ExprKind::Mac(..) => return self.visit_macro_invoc(expr.id, false),
            ExprKind::Repeat(_, ref count) => self.visit_const_expr(count),
            ExprKind::Closure(..) => {
                let def = self.create_def(expr.id,
                                          DefPathData::ClosureExpr,
                                          REGULAR_SPACE);
                self.parent_def = Some(def);
            }
            _ => {}
        }

        visit::walk_expr(self, expr);
        self.parent_def = parent_def;
    }

    fn visit_ty(&mut self, ty: &'a Ty) {
        match ty.node {
            TyKind::Mac(..) => return self.visit_macro_invoc(ty.id, false),
            TyKind::Array(_, ref length) => self.visit_const_expr(length),
            TyKind::ImplTrait(..) => {
                self.create_def(ty.id, DefPathData::ImplTrait, REGULAR_SPACE);
            }
            TyKind::Typeof(ref expr) => self.visit_const_expr(expr),
            _ => {}
        }
        visit::walk_ty(self, ty);
    }

    fn visit_stmt(&mut self, stmt: &'a Stmt) {
        match stmt.node {
            StmtKind::Mac(..) => self.visit_macro_invoc(stmt.id, false),
            _ => visit::walk_stmt(self, stmt),
        }
    }

    fn visit_token(&mut self, t: Token) {
        if let Token::Interpolated(nt) = t {
            match nt.0 {
                token::NtExpr(ref expr) => {
                    if let ExprKind::Mac(..) = expr.node {
                        self.visit_macro_invoc(expr.id, false);
                    }
                }
                _ => {}
            }
        }
    }
}
