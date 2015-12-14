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
use rustc_front::intravisit::{self, Visitor};
use middle::def_id::{CRATE_DEF_INDEX, DefIndex};
use std::iter::repeat;
use syntax::ast::{NodeId, CRATE_NODE_ID, DUMMY_NODE_ID};
use syntax::codemap::Span;

/// A Visitor that walks over an AST and collects Node's into an AST
/// Map.
pub struct NodeCollector<'ast> {
    pub krate: &'ast Crate,
    pub map: Vec<MapEntry<'ast>>,
    pub definitions: Definitions,
    pub parent_node: NodeId,
}

impl<'ast> NodeCollector<'ast> {
    pub fn root(krate: &'ast Crate) -> NodeCollector<'ast> {
        let mut collector = NodeCollector {
            krate: krate,
            map: vec![],
            definitions: Definitions::new(),
            parent_node: CRATE_NODE_ID,
        };
        collector.insert_entry(CRATE_NODE_ID, RootCrate);

        let result = collector.create_def_with_parent(None, CRATE_NODE_ID, DefPathData::CrateRoot);
        assert_eq!(result, CRATE_DEF_INDEX);

        collector.create_def_with_parent(Some(CRATE_DEF_INDEX), DUMMY_NODE_ID, DefPathData::Misc);

        collector
    }

    pub fn extend(krate: &'ast Crate,
                  parent: &'ast InlinedParent,
                  parent_node: NodeId,
                  parent_def_path: DefPath,
                  map: Vec<MapEntry<'ast>>,
                  definitions: Definitions)
                  -> NodeCollector<'ast> {
        let mut collector = NodeCollector {
            krate: krate,
            map: map,
            parent_node: parent_node,
            definitions: definitions,
        };

        collector.insert_entry(parent_node, RootInlinedParent(parent));
        collector.create_def(parent_node, DefPathData::InlinedRoot(parent_def_path));

        collector
    }

    fn parent_def(&self) -> Option<DefIndex> {
        let mut parent_node = Some(self.parent_node);
        while let Some(p) = parent_node {
            if let Some(q) = self.definitions.opt_def_index(p) {
                return Some(q);
            }
            parent_node = self.map[p as usize].parent_node();
        }
        None
    }

    fn create_def(&mut self, node_id: NodeId, data: DefPathData) -> DefIndex {
        let parent_def = self.parent_def();
        self.definitions.create_def_with_parent(parent_def, node_id, data)
    }

    fn create_def_with_parent(&mut self,
                              parent: Option<DefIndex>,
                              node_id: NodeId,
                              data: DefPathData)
                              -> DefIndex {
        self.definitions.create_def_with_parent(parent, node_id, data)
    }

    fn insert_entry(&mut self, id: NodeId, entry: MapEntry<'ast>) {
        debug!("ast_map: {:?} => {:?}", id, entry);
        let len = self.map.len();
        if id as usize >= len {
            self.map.extend(repeat(NotPresent).take(id as usize - len + 1));
        }
        self.map[id as usize] = entry;
    }

    fn insert_def(&mut self, id: NodeId, node: Node<'ast>, data: DefPathData) -> DefIndex {
        self.insert(id, node);
        self.create_def(id, data)
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
        self.visit_item(self.krate.item(item.id))
    }

    fn visit_item(&mut self, i: &'ast Item) {
        // Pick the def data. This need not be unique, but the more
        // information we encapsulate into
        let def_data = match i.node {
            ItemDefaultImpl(..) | ItemImpl(..) => DefPathData::Impl(i.name),
            ItemEnum(..) | ItemStruct(..) | ItemTrait(..) => DefPathData::Type(i.name),
            ItemExternCrate(..) | ItemMod(..) => DefPathData::Mod(i.name),
            ItemStatic(..) | ItemConst(..) | ItemFn(..) => DefPathData::Value(i.name),
            _ => DefPathData::Misc,
        };

        self.insert_def(i.id, NodeItem(i), def_data);

        let parent_node = self.parent_node;
        self.parent_node = i.id;

        match i.node {
            ItemImpl(..) => {}
            ItemEnum(ref enum_definition, _) => {
                for v in &enum_definition.variants {
                    let variant_def_index =
                        self.insert_def(v.node.data.id(),
                                        NodeVariant(v),
                                        DefPathData::EnumVariant(v.node.name));

                    for field in v.node.data.fields() {
                        self.create_def_with_parent(
                            Some(variant_def_index),
                            field.node.id,
                            DefPathData::Field(field.node.kind));
                    }
                }
            }
            ItemForeignMod(..) => {
            }
            ItemStruct(ref struct_def, _) => {
                // If this is a tuple-like struct, register the constructor.
                if !struct_def.is_struct() {
                    self.insert_def(struct_def.id(),
                                    NodeStructCtor(struct_def),
                                    DefPathData::StructCtor);
                }

                for field in struct_def.fields() {
                    self.create_def(field.node.id, DefPathData::Field(field.node.kind));
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
        self.insert_def(foreign_item.id,
                        NodeForeignItem(foreign_item),
                        DefPathData::Value(foreign_item.name));

        let parent_node = self.parent_node;
        self.parent_node = foreign_item.id;
        intravisit::walk_foreign_item(self, foreign_item);
        self.parent_node = parent_node;
    }

    fn visit_generics(&mut self, generics: &'ast Generics) {
        for ty_param in generics.ty_params.iter() {
            self.insert_def(ty_param.id,
                            NodeTyParam(ty_param),
                            DefPathData::TypeParam(ty_param.name));
        }

        intravisit::walk_generics(self, generics);
    }

    fn visit_trait_item(&mut self, ti: &'ast TraitItem) {
        let def_data = match ti.node {
            MethodTraitItem(..) | ConstTraitItem(..) => DefPathData::Value(ti.name),
            TypeTraitItem(..) => DefPathData::Type(ti.name),
        };

        self.insert(ti.id, NodeTraitItem(ti));
        self.create_def(ti.id, def_data);

        let parent_node = self.parent_node;
        self.parent_node = ti.id;

        match ti.node {
            ConstTraitItem(_, Some(ref expr)) => {
                self.create_def(expr.id, DefPathData::Initializer);
            }
            _ => { }
        }

        intravisit::walk_trait_item(self, ti);

        self.parent_node = parent_node;
    }

    fn visit_impl_item(&mut self, ii: &'ast ImplItem) {
        let def_data = match ii.node {
            ImplItemKind::Method(..) | ImplItemKind::Const(..) => DefPathData::Value(ii.name),
            ImplItemKind::Type(..) => DefPathData::Type(ii.name),
        };

        self.insert_def(ii.id, NodeImplItem(ii), def_data);

        let parent_node = self.parent_node;
        self.parent_node = ii.id;

        match ii.node {
            ImplItemKind::Const(_, ref expr) => {
                self.create_def(expr.id, DefPathData::Initializer);
            }
            _ => { }
        }

        intravisit::walk_impl_item(self, ii);

        self.parent_node = parent_node;
    }

    fn visit_pat(&mut self, pat: &'ast Pat) {
        let maybe_binding = match pat.node {
            PatIdent(_, id, _) => Some(id.node),
            _ => None
        };

        if let Some(id) = maybe_binding {
            self.insert_def(pat.id, NodeLocal(pat), DefPathData::Binding(id.name));
        } else {
            self.insert(pat.id, NodePat(pat));
        }

        let parent_node = self.parent_node;
        self.parent_node = pat.id;
        intravisit::walk_pat(self, pat);
        self.parent_node = parent_node;
    }

    fn visit_expr(&mut self, expr: &'ast Expr) {
        self.insert(expr.id, NodeExpr(expr));

        match expr.node {
            ExprClosure(..) => { self.create_def(expr.id, DefPathData::ClosureExpr); }
            _ => { }
        }

        let parent_node = self.parent_node;
        self.parent_node = expr.id;
        intravisit::walk_expr(self, expr);
        self.parent_node = parent_node;
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt) {
        let id = util::stmt_id(stmt);
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

    fn visit_lifetime_def(&mut self, def: &'ast LifetimeDef) {
        self.create_def(def.lifetime.id, DefPathData::LifetimeDef(def.lifetime.name));
        self.visit_lifetime(&def.lifetime);
    }

    fn visit_macro_def(&mut self, macro_def: &'ast MacroDef) {
        self.create_def(macro_def.id, DefPathData::MacroDef(macro_def.name));
    }
}
