// Copyright 2015-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::*;

use hir::intravisit::{Visitor, NestedVisitorMap};
use std::iter::repeat;
use syntax::ast::{NodeId, CRATE_NODE_ID};
use syntax_pos::Span;

/// A Visitor that walks over the HIR and collects Nodes into a HIR map
pub struct NodeCollector<'ast> {
    /// The crate
    pub krate: &'ast Crate,
    /// The node map
    pub(super) map: Vec<MapEntry<'ast>>,
    /// The parent of this node
    pub parent_node: NodeId,
}

impl<'ast> NodeCollector<'ast> {
    pub fn root(krate: &'ast Crate) -> NodeCollector<'ast> {
        let mut collector = NodeCollector {
            krate: krate,
            map: vec![],
            parent_node: CRATE_NODE_ID,
        };
        collector.insert_entry(CRATE_NODE_ID, RootCrate);

        collector
    }

    fn insert_entry(&mut self, id: NodeId, entry: MapEntry<'ast>) {
        debug!("ast_map: {:?} => {:?}", id, entry);
        let len = self.map.len();
        if id.as_usize() >= len {
            self.map.extend(repeat(NotPresent).take(id.as_usize() - len + 1));
        }
        self.map[id.as_usize()] = entry;
    }

    fn insert(&mut self, id: NodeId, node: Node<'ast>) {
        let entry = MapEntry::from_node(self.parent_node, node);
        self.insert_entry(id, entry);
    }

    fn with_parent<F: FnOnce(&mut Self)>(&mut self, parent_id: NodeId, f: F) {
        let parent_node = self.parent_node;
        self.parent_node = parent_id;
        f(self);
        self.parent_node = parent_node;
    }
}

impl<'ast> Visitor<'ast> for NodeCollector<'ast> {
    /// Because we want to track parent items and so forth, enable
    /// deep walking so that we walk nested items in the context of
    /// their outer items.

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'ast> {
        panic!("visit_nested_xxx must be manually implemented in this visitor")
    }

    fn visit_nested_item(&mut self, item: ItemId) {
        debug!("visit_nested_item: {:?}", item);
        self.visit_item(self.krate.item(item.id));
    }

    fn visit_nested_trait_item(&mut self, item_id: TraitItemId) {
        self.visit_trait_item(self.krate.trait_item(item_id));
    }

    fn visit_nested_impl_item(&mut self, item_id: ImplItemId) {
        self.visit_impl_item(self.krate.impl_item(item_id));
    }

    fn visit_nested_body(&mut self, id: BodyId) {
        self.visit_body(self.krate.body(id));
    }

    fn visit_item(&mut self, i: &'ast Item) {
        debug!("visit_item: {:?}", i);

        self.insert(i.id, NodeItem(i));

        self.with_parent(i.id, |this| {
            match i.node {
                ItemStruct(ref struct_def, _) => {
                    // If this is a tuple-like struct, register the constructor.
                    if !struct_def.is_struct() {
                        this.insert(struct_def.id(), NodeStructCtor(struct_def));
                    }
                }
                _ => {}
            }
            intravisit::walk_item(this, i);
        });
    }

    fn visit_foreign_item(&mut self, foreign_item: &'ast ForeignItem) {
        self.insert(foreign_item.id, NodeForeignItem(foreign_item));

        self.with_parent(foreign_item.id, |this| {
            intravisit::walk_foreign_item(this, foreign_item);
        });
    }

    fn visit_generics(&mut self, generics: &'ast Generics) {
        for ty_param in generics.ty_params.iter() {
            self.insert(ty_param.id, NodeTyParam(ty_param));
        }

        intravisit::walk_generics(self, generics);
    }

    fn visit_trait_item(&mut self, ti: &'ast TraitItem) {
        self.insert(ti.id, NodeTraitItem(ti));

        self.with_parent(ti.id, |this| {
            intravisit::walk_trait_item(this, ti);
        });
    }

    fn visit_impl_item(&mut self, ii: &'ast ImplItem) {
        self.insert(ii.id, NodeImplItem(ii));

        self.with_parent(ii.id, |this| {
            intravisit::walk_impl_item(this, ii);
        });
    }

    fn visit_pat(&mut self, pat: &'ast Pat) {
        let node = if let PatKind::Binding(..) = pat.node {
            NodeLocal(pat)
        } else {
            NodePat(pat)
        };
        self.insert(pat.id, node);

        self.with_parent(pat.id, |this| {
            intravisit::walk_pat(this, pat);
        });
    }

    fn visit_expr(&mut self, expr: &'ast Expr) {
        self.insert(expr.id, NodeExpr(expr));

        self.with_parent(expr.id, |this| {
            intravisit::walk_expr(this, expr);
        });
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt) {
        let id = stmt.node.id();
        self.insert(id, NodeStmt(stmt));

        self.with_parent(id, |this| {
            intravisit::walk_stmt(this, stmt);
        });
    }

    fn visit_ty(&mut self, ty: &'ast Ty) {
        self.insert(ty.id, NodeTy(ty));

        self.with_parent(ty.id, |this| {
            intravisit::walk_ty(this, ty);
        });
    }

    fn visit_trait_ref(&mut self, tr: &'ast TraitRef) {
        self.insert(tr.ref_id, NodeTraitRef(tr));

        self.with_parent(tr.ref_id, |this| {
            intravisit::walk_trait_ref(this, tr);
        });
    }

    fn visit_fn(&mut self, fk: intravisit::FnKind<'ast>, fd: &'ast FnDecl,
                b: BodyId, s: Span, id: NodeId) {
        assert_eq!(self.parent_node, id);
        intravisit::walk_fn(self, fk, fd, b, s, id);
    }

    fn visit_block(&mut self, block: &'ast Block) {
        self.insert(block.id, NodeBlock(block));
        self.with_parent(block.id, |this| {
            intravisit::walk_block(this, block);
        });
    }

    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime) {
        self.insert(lifetime.id, NodeLifetime(lifetime));
    }

    fn visit_vis(&mut self, visibility: &'ast Visibility) {
        match *visibility {
            Visibility::Public |
            Visibility::Crate |
            Visibility::Inherited => {}
            Visibility::Restricted { id, .. } => {
                self.insert(id, NodeVisibility(visibility));
                self.with_parent(id, |this| {
                    intravisit::walk_vis(this, visibility);
                });
            }
        }
    }

    fn visit_macro_def(&mut self, macro_def: &'ast MacroDef) {
        self.insert_entry(macro_def.id, NotPresent);
    }

    fn visit_variant(&mut self, v: &'ast Variant, g: &'ast Generics, item_id: NodeId) {
        let id = v.node.data.id();
        self.insert(id, NodeVariant(v));
        self.with_parent(id, |this| {
            intravisit::walk_variant(this, v, g, item_id);
        });
    }

    fn visit_struct_field(&mut self, field: &'ast StructField) {
        self.insert(field.id, NodeField(field));
        self.with_parent(field.id, |this| {
            intravisit::walk_struct_field(this, field);
        });
    }
}
