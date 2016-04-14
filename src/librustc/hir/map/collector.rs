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
use super::MapEntry::*;

use hir::*;
use hir::intravisit::Visitor;
use hir::def_id::DefId;
use middle::cstore::InlinedItem;
use std::iter::repeat;
use syntax::ast::{NodeId, CRATE_NODE_ID};
use syntax::codemap::Span;

/// A Visitor that walks over the HIR and collects Node's into a HIR map.
pub struct NodeCollector<'ast> {
    pub krate: &'ast Crate,
    pub map: Vec<MapEntry<'ast>>,
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

    pub fn extend(krate: &'ast Crate,
                  parent: &'ast InlinedItem,
                  parent_node: NodeId,
                  parent_def_path: DefPath,
                  parent_def_id: DefId,
                  map: Vec<MapEntry<'ast>>)
                  -> NodeCollector<'ast> {
        let mut collector = NodeCollector {
            krate: krate,
            map: map,
            parent_node: parent_node,
        };

        assert_eq!(parent_def_path.krate, parent_def_id.krate);
        collector.insert_entry(parent_node, RootInlinedParent(parent));

        collector
    }

    fn insert_entry(&mut self, id: NodeId, entry: MapEntry<'ast>) {
        debug!("ast_map: {:?} => {:?}", id, entry);
        let len = self.map.len();
        if id as usize >= len {
            self.map.extend(repeat(NotPresent).take(id as usize - len + 1));
        }
        self.map[id as usize] = entry;
    }

    fn insert(&mut self, id: NodeId, node: Node<'ast>) {
        let entry = MapEntry::from_node(self.parent_node, node);
        self.insert_entry(id, entry);
    }
}

impl<'ast> Visitor<'ast> for NodeCollector<'ast> {
    /// Because we want to track parent items and so forth, enable
    /// deep walking so that we walk nested items in the context of
    /// their outer items.
    fn visit_nested_item(&mut self, item: ItemId) {
        debug!("visit_nested_item: {:?}", item);
        self.visit_item(self.krate.item(item.id))
    }

    fn visit_item(&mut self, i: &'ast Item) {
        debug!("visit_item: {:?}", i);

        self.insert(i.id, NodeItem(i));

        let parent_node = self.parent_node;
        self.parent_node = i.id;

        match i.node {
            ItemEnum(ref enum_definition, _) => {
                for v in &enum_definition.variants {
                    self.insert(v.node.data.id(), NodeVariant(v));
                }
            }
            ItemStruct(ref struct_def, _) => {
                // If this is a tuple-like struct, register the constructor.
                if !struct_def.is_struct() {
                    self.insert(struct_def.id(), NodeStructCtor(struct_def));
                }
            }
            ItemTrait(_, _, ref bounds, _) => {
                for b in bounds.iter() {
                    if let TraitTyParamBound(ref t, TraitBoundModifier::None) = *b {
                        self.insert(t.trait_ref.ref_id, NodeItem(i));
                    }
                }
            }
            ItemUse(ref view_path) => {
                match view_path.node {
                    ViewPathList(_, ref paths) => {
                        for path in paths {
                            self.insert(path.node.id(), NodeItem(i));
                        }
                    }
                    _ => ()
                }
            }
            _ => {}
        }
        intravisit::walk_item(self, i);
        self.parent_node = parent_node;
    }

    fn visit_foreign_item(&mut self, foreign_item: &'ast ForeignItem) {
        self.insert(foreign_item.id, NodeForeignItem(foreign_item));

        let parent_node = self.parent_node;
        self.parent_node = foreign_item.id;
        intravisit::walk_foreign_item(self, foreign_item);
        self.parent_node = parent_node;
    }

    fn visit_generics(&mut self, generics: &'ast Generics) {
        for ty_param in generics.ty_params.iter() {
            self.insert(ty_param.id, NodeTyParam(ty_param));
        }

        intravisit::walk_generics(self, generics);
    }

    fn visit_trait_item(&mut self, ti: &'ast TraitItem) {
        self.insert(ti.id, NodeTraitItem(ti));

        let parent_node = self.parent_node;
        self.parent_node = ti.id;

        intravisit::walk_trait_item(self, ti);

        self.parent_node = parent_node;
    }

    fn visit_impl_item(&mut self, ii: &'ast ImplItem) {
        self.insert(ii.id, NodeImplItem(ii));

        let parent_node = self.parent_node;
        self.parent_node = ii.id;

        intravisit::walk_impl_item(self, ii);

        self.parent_node = parent_node;
    }

    fn visit_pat(&mut self, pat: &'ast Pat) {
        self.insert(pat.id, NodeLocal(pat));

        let parent_node = self.parent_node;
        self.parent_node = pat.id;
        intravisit::walk_pat(self, pat);
        self.parent_node = parent_node;
    }

    fn visit_expr(&mut self, expr: &'ast Expr) {
        self.insert(expr.id, NodeExpr(expr));

        let parent_node = self.parent_node;
        self.parent_node = expr.id;
        intravisit::walk_expr(self, expr);
        self.parent_node = parent_node;
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt) {
        let id = stmt.node.id();
        self.insert(id, NodeStmt(stmt));
        let parent_node = self.parent_node;
        self.parent_node = id;
        intravisit::walk_stmt(self, stmt);
        self.parent_node = parent_node;
    }

    fn visit_fn(&mut self, fk: intravisit::FnKind<'ast>, fd: &'ast FnDecl,
                b: &'ast Block, s: Span, id: NodeId) {
        assert_eq!(self.parent_node, id);
        intravisit::walk_fn(self, fk, fd, b, s);
    }

    fn visit_block(&mut self, block: &'ast Block) {
        self.insert(block.id, NodeBlock(block));
        let parent_node = self.parent_node;
        self.parent_node = block.id;
        intravisit::walk_block(self, block);
        self.parent_node = parent_node;
    }

    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime) {
        self.insert(lifetime.id, NodeLifetime(lifetime));
    }
}
