// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
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

use rustc_front::hir::*;
use rustc_front::util;
use rustc_front::visit::{self, Visitor};
use std::iter::repeat;
use syntax::ast::{NodeId, CRATE_NODE_ID, DUMMY_NODE_ID};
use syntax::codemap::Span;
use util::nodemap::NodeSet;

/// A Visitor that walks over an AST and collects Node's into an AST
/// Map.
pub struct NodeCollector<'ast> {
    pub map: Vec<MapEntry<'ast>>,
    pub definitions_map: NodeSet,
    pub parent_node: NodeId,
}

impl<'ast> NodeCollector<'ast> {
    pub fn root() -> NodeCollector<'ast> {
        let mut collector = NodeCollector {
            map: vec![],
            definitions_map: NodeSet(),
            parent_node: CRATE_NODE_ID,
        };
        collector.insert_entry(CRATE_NODE_ID, RootCrate);
        collector.create_def(CRATE_NODE_ID);
        collector.create_def(DUMMY_NODE_ID);
        collector
    }

    pub fn extend(parent: &'ast InlinedParent,
                  parent_node: NodeId,
                  map: Vec<MapEntry<'ast>>,
                  definitions_map: NodeSet)
                  -> NodeCollector<'ast> {
        let mut collector = NodeCollector {
            map: map,
            definitions_map: definitions_map,
            parent_node: parent_node
        };
        collector.insert_entry(parent_node, RootInlinedParent(parent));

        collector
    }

    fn create_def(&mut self, node: NodeId) {
        let is_new = self.definitions_map.insert(node);
        assert!(is_new,
                "two entries for node id `{}` -- previous is `{:?}`",
                node, node);
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

    fn visit_fn_decl(&mut self, decl: &'ast FnDecl) {
        for a in &decl.inputs {
            self.insert(a.id, NodeArg(&*a.pat));
        }
    }
}

impl<'ast> Visitor<'ast> for NodeCollector<'ast> {
    fn visit_item(&mut self, i: &'ast Item) {
        self.insert(i.id, NodeItem(i));

        let parent_node = self.parent_node;
        self.parent_node = i.id;

        self.create_def(i.id);

        match i.node {
            ItemImpl(..) => { }
            ItemEnum(ref enum_definition, _) => {
                for v in &enum_definition.variants {
                    self.insert(v.node.id, NodeVariant(&**v));
                    self.create_def(v.node.id);

                    match v.node.kind {
                        TupleVariantKind(ref args) => {
                            for arg in args {
                                self.create_def(arg.id);
                            }
                        }
                        StructVariantKind(ref def) => {
                            for field in &def.fields {
                                self.create_def(field.node.id);
                            }
                        }
                    }
                }
            }
            ItemForeignMod(..) => {}
            ItemStruct(ref struct_def, _) => {
                // If this is a tuple-like struct, register the constructor.
                match struct_def.ctor_id {
                    Some(ctor_id) => {
                        self.insert(ctor_id, NodeStructCtor(&**struct_def));
                        self.create_def(ctor_id);
                    }
                    None => {}
                }

                for field in &struct_def.fields {
                    self.create_def(field.node.id);
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
        visit::walk_item(self, i);
        self.parent_node = parent_node;
    }

    fn visit_foreign_item(&mut self, foreign_item: &'ast ForeignItem) {
        self.insert(foreign_item.id, NodeForeignItem(foreign_item));
        self.create_def(foreign_item.id);

        let parent_node = self.parent_node;
        self.parent_node = foreign_item.id;
        visit::walk_foreign_item(self, foreign_item);
        self.parent_node = parent_node;
    }

    fn visit_generics(&mut self, generics: &'ast Generics) {
        for ty_param in generics.ty_params.iter() {
            self.create_def(ty_param.id);
            self.insert(ty_param.id, NodeTyParam(ty_param));
        }

        visit::walk_generics(self, generics);
    }

    fn visit_trait_item(&mut self, ti: &'ast TraitItem) {
        self.insert(ti.id, NodeTraitItem(ti));
        self.create_def(ti.id);

        match ti.node {
            ConstTraitItem(_, Some(ref expr)) => {
                self.create_def(expr.id);
            }
            _ => { }
        }

        let parent_node = self.parent_node;
        self.parent_node = ti.id;
        visit::walk_trait_item(self, ti);
        self.parent_node = parent_node;
    }

    fn visit_impl_item(&mut self, ii: &'ast ImplItem) {
        self.insert(ii.id, NodeImplItem(ii));
        self.create_def(ii.id);

        match ii.node {
            ConstImplItem(_, ref expr) => {
                self.create_def(expr.id);
            }
            _ => { }
        }

        let parent_node = self.parent_node;
        self.parent_node = ii.id;
        visit::walk_impl_item(self, ii);
        self.parent_node = parent_node;
    }

    fn visit_pat(&mut self, pat: &'ast Pat) {
        let maybe_binding = match pat.node {
            PatIdent(..) => true,
            _ => false
        };

        self.insert(pat.id,
                    if maybe_binding {NodeLocal(pat)} else {NodePat(pat)});

        if maybe_binding {
            self.create_def(pat.id);
        }

        let parent_node = self.parent_node;
        self.parent_node = pat.id;
        visit::walk_pat(self, pat);
        self.parent_node = parent_node;
    }

    fn visit_expr(&mut self, expr: &'ast Expr) {
        self.insert(expr.id, NodeExpr(expr));

        match expr.node {
            ExprClosure(..) => self.create_def(expr.id),
            _ => (),
        }

        let parent_node = self.parent_node;
        self.parent_node = expr.id;
        visit::walk_expr(self, expr);
        self.parent_node = parent_node;
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt) {
        let id = util::stmt_id(stmt);
        self.insert(id, NodeStmt(stmt));
        let parent_node = self.parent_node;
        self.parent_node = id;
        visit::walk_stmt(self, stmt);
        self.parent_node = parent_node;
    }

    fn visit_fn(&mut self, fk: visit::FnKind<'ast>, fd: &'ast FnDecl,
                b: &'ast Block, s: Span, id: NodeId) {
        let parent_node = self.parent_node;
        self.parent_node = id;
        self.visit_fn_decl(fd);
        visit::walk_fn(self, fk, fd, b, s);
        self.parent_node = parent_node;
    }

    fn visit_ty(&mut self, ty: &'ast Ty) {
        let parent_node = self.parent_node;
        self.parent_node = ty.id;
        match ty.node {
            TyBareFn(ref fd) => {
                self.visit_fn_decl(&*fd.decl);
            }
            _ => {}
        }
        visit::walk_ty(self, ty);
        self.parent_node = parent_node;
    }

    fn visit_block(&mut self, block: &'ast Block) {
        self.insert(block.id, NodeBlock(block));
        let parent_node = self.parent_node;
        self.parent_node = block.id;
        visit::walk_block(self, block);
        self.parent_node = parent_node;
    }

    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime) {
        self.insert(lifetime.id, NodeLifetime(lifetime));
    }

    fn visit_lifetime_def(&mut self, def: &'ast LifetimeDef) {
        self.create_def(def.lifetime.id);
        self.visit_lifetime(&def.lifetime);
    }

    fn visit_macro_def(&mut self, macro_def: &'ast MacroDef) {
        self.create_def(macro_def.id);
    }
}

