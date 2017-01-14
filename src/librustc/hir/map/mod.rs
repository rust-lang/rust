// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::Node::*;
use self::MapEntry::*;
use self::collector::NodeCollector;
pub use self::def_collector::{DefCollector, MacroInvocationData};
pub use self::definitions::{Definitions, DefKey, DefPath, DefPathData,
                            DisambiguatedDefPathData};

use dep_graph::{DepGraph, DepNode};

use hir::def_id::{CRATE_DEF_INDEX, DefId, DefIndex};

use syntax::abi::Abi;
use syntax::ast::{self, Name, NodeId, CRATE_NODE_ID};
use syntax::codemap::Spanned;
use syntax_pos::Span;

use hir::*;
use hir::print::Nested;
use util::nodemap::DefIdMap;

use arena::TypedArena;
use std::cell::RefCell;
use std::io;

pub mod blocks;
mod collector;
mod def_collector;
pub mod definitions;

#[derive(Copy, Clone, Debug)]
pub enum Node<'ast> {
    NodeItem(&'ast Item),
    NodeForeignItem(&'ast ForeignItem),
    NodeTraitItem(&'ast TraitItem),
    NodeImplItem(&'ast ImplItem),
    NodeVariant(&'ast Variant),
    NodeField(&'ast StructField),
    NodeExpr(&'ast Expr),
    NodeStmt(&'ast Stmt),
    NodeTy(&'ast Ty),
    NodeTraitRef(&'ast TraitRef),
    NodeLocal(&'ast Pat),
    NodePat(&'ast Pat),
    NodeBlock(&'ast Block),

    /// NodeStructCtor represents a tuple struct.
    NodeStructCtor(&'ast VariantData),

    NodeLifetime(&'ast Lifetime),
    NodeTyParam(&'ast TyParam),
    NodeVisibility(&'ast Visibility),
}

/// Represents an entry and its parent NodeID.
/// The odd layout is to bring down the total size.
#[derive(Copy, Debug)]
enum MapEntry<'ast> {
    /// Placeholder for holes in the map.
    NotPresent,

    /// All the node types, with a parent ID.
    EntryItem(NodeId, &'ast Item),
    EntryForeignItem(NodeId, &'ast ForeignItem),
    EntryTraitItem(NodeId, &'ast TraitItem),
    EntryImplItem(NodeId, &'ast ImplItem),
    EntryVariant(NodeId, &'ast Variant),
    EntryField(NodeId, &'ast StructField),
    EntryExpr(NodeId, &'ast Expr),
    EntryStmt(NodeId, &'ast Stmt),
    EntryTy(NodeId, &'ast Ty),
    EntryTraitRef(NodeId, &'ast TraitRef),
    EntryLocal(NodeId, &'ast Pat),
    EntryPat(NodeId, &'ast Pat),
    EntryBlock(NodeId, &'ast Block),
    EntryStructCtor(NodeId, &'ast VariantData),
    EntryLifetime(NodeId, &'ast Lifetime),
    EntryTyParam(NodeId, &'ast TyParam),
    EntryVisibility(NodeId, &'ast Visibility),

    /// Roots for node trees.
    RootCrate,
}

impl<'ast> Clone for MapEntry<'ast> {
    fn clone(&self) -> MapEntry<'ast> {
        *self
    }
}

impl<'ast> MapEntry<'ast> {
    fn from_node(p: NodeId, node: Node<'ast>) -> MapEntry<'ast> {
        match node {
            NodeItem(n) => EntryItem(p, n),
            NodeForeignItem(n) => EntryForeignItem(p, n),
            NodeTraitItem(n) => EntryTraitItem(p, n),
            NodeImplItem(n) => EntryImplItem(p, n),
            NodeVariant(n) => EntryVariant(p, n),
            NodeField(n) => EntryField(p, n),
            NodeExpr(n) => EntryExpr(p, n),
            NodeStmt(n) => EntryStmt(p, n),
            NodeTy(n) => EntryTy(p, n),
            NodeTraitRef(n) => EntryTraitRef(p, n),
            NodeLocal(n) => EntryLocal(p, n),
            NodePat(n) => EntryPat(p, n),
            NodeBlock(n) => EntryBlock(p, n),
            NodeStructCtor(n) => EntryStructCtor(p, n),
            NodeLifetime(n) => EntryLifetime(p, n),
            NodeTyParam(n) => EntryTyParam(p, n),
            NodeVisibility(n) => EntryVisibility(p, n),
        }
    }

    fn parent_node(self) -> Option<NodeId> {
        Some(match self {
            EntryItem(id, _) => id,
            EntryForeignItem(id, _) => id,
            EntryTraitItem(id, _) => id,
            EntryImplItem(id, _) => id,
            EntryVariant(id, _) => id,
            EntryField(id, _) => id,
            EntryExpr(id, _) => id,
            EntryStmt(id, _) => id,
            EntryTy(id, _) => id,
            EntryTraitRef(id, _) => id,
            EntryLocal(id, _) => id,
            EntryPat(id, _) => id,
            EntryBlock(id, _) => id,
            EntryStructCtor(id, _) => id,
            EntryLifetime(id, _) => id,
            EntryTyParam(id, _) => id,
            EntryVisibility(id, _) => id,

            NotPresent |
            RootCrate => return None,
        })
    }

    fn to_node(self) -> Option<Node<'ast>> {
        Some(match self {
            EntryItem(_, n) => NodeItem(n),
            EntryForeignItem(_, n) => NodeForeignItem(n),
            EntryTraitItem(_, n) => NodeTraitItem(n),
            EntryImplItem(_, n) => NodeImplItem(n),
            EntryVariant(_, n) => NodeVariant(n),
            EntryField(_, n) => NodeField(n),
            EntryExpr(_, n) => NodeExpr(n),
            EntryStmt(_, n) => NodeStmt(n),
            EntryTy(_, n) => NodeTy(n),
            EntryTraitRef(_, n) => NodeTraitRef(n),
            EntryLocal(_, n) => NodeLocal(n),
            EntryPat(_, n) => NodePat(n),
            EntryBlock(_, n) => NodeBlock(n),
            EntryStructCtor(_, n) => NodeStructCtor(n),
            EntryLifetime(_, n) => NodeLifetime(n),
            EntryTyParam(_, n) => NodeTyParam(n),
            EntryVisibility(_, n) => NodeVisibility(n),
            _ => return None
        })
    }

    fn is_body_owner(self, node_id: NodeId) -> bool {
        match self {
            EntryItem(_, item) => {
                match item.node {
                    ItemConst(_, body) |
                    ItemStatic(.., body) |
                    ItemFn(_, _, _, _, _, body) => body.node_id == node_id,
                    _ => false
                }
            }

            EntryTraitItem(_, item) => {
                match item.node {
                    TraitItemKind::Const(_, Some(body)) |
                    TraitItemKind::Method(_, TraitMethod::Provided(body)) => {
                        body.node_id == node_id
                    }
                    _ => false
                }
            }

            EntryImplItem(_, item) => {
                match item.node {
                    ImplItemKind::Const(_, body) |
                    ImplItemKind::Method(_, body) => body.node_id == node_id,
                    _ => false
                }
            }

            EntryExpr(_, expr) => {
                match expr.node {
                    ExprClosure(.., body, _) => body.node_id == node_id,
                    _ => false
                }
            }

            _ => false
        }
    }
}

/// Stores a crate and any number of inlined items from other crates.
pub struct Forest {
    krate: Crate,
    pub dep_graph: DepGraph,
    inlined_bodies: TypedArena<Body>
}

impl Forest {
    pub fn new(krate: Crate, dep_graph: &DepGraph) -> Forest {
        Forest {
            krate: krate,
            dep_graph: dep_graph.clone(),
            inlined_bodies: TypedArena::new()
        }
    }

    pub fn krate<'ast>(&'ast self) -> &'ast Crate {
        self.dep_graph.read(DepNode::Krate);
        &self.krate
    }
}

/// Represents a mapping from Node IDs to AST elements and their parent
/// Node IDs
#[derive(Clone)]
pub struct Map<'ast> {
    /// The backing storage for all the AST nodes.
    pub forest: &'ast Forest,

    /// Same as the dep_graph in forest, just available with one fewer
    /// deref. This is a gratuitious micro-optimization.
    pub dep_graph: DepGraph,

    /// NodeIds are sequential integers from 0, so we can be
    /// super-compact by storing them in a vector. Not everything with
    /// a NodeId is in the map, but empirically the occupancy is about
    /// 75-80%, so there's not too much overhead (certainly less than
    /// a hashmap, since they (at the time of writing) have a maximum
    /// of 75% occupancy).
    ///
    /// Also, indexing is pretty quick when you've got a vector and
    /// plain old integers.
    map: Vec<MapEntry<'ast>>,

    definitions: Definitions,

    /// Bodies inlined from other crates are cached here.
    inlined_bodies: RefCell<DefIdMap<&'ast Body>>,
}

impl<'ast> Map<'ast> {
    /// Registers a read in the dependency graph of the AST node with
    /// the given `id`. This needs to be called each time a public
    /// function returns the HIR for a node -- in other words, when it
    /// "reveals" the content of a node to the caller (who might not
    /// otherwise have had access to those contents, and hence needs a
    /// read recorded). If the function just returns a DefId or
    /// NodeId, no actual content was returned, so no read is needed.
    pub fn read(&self, id: NodeId) {
        self.dep_graph.read(self.dep_node(id));
    }

    fn dep_node(&self, id0: NodeId) -> DepNode<DefId> {
        let mut id = id0;
        let mut last_expr = None;
        loop {
            let entry = self.map[id.as_usize()];
            match entry {
                EntryItem(..) |
                EntryTraitItem(..) |
                EntryImplItem(..) => {
                    if let Some(last_id) = last_expr {
                        // The body may have a separate dep node
                        if entry.is_body_owner(last_id) {
                            let def_id = self.local_def_id(id);
                            return DepNode::HirBody(def_id);
                        }
                    }
                    return DepNode::Hir(self.local_def_id(id));
                }

                EntryVariant(p, v) => {
                    id = p;

                    if last_expr.is_some() {
                        if v.node.disr_expr.map(|e| e.node_id) == last_expr {
                            // The enum parent holds both Hir and HirBody nodes.
                            let def_id = self.local_def_id(id);
                            return DepNode::HirBody(def_id);
                        }
                    }
                }

                EntryForeignItem(p, _) |
                EntryField(p, _) |
                EntryStmt(p, _) |
                EntryTy(p, _) |
                EntryTraitRef(p, _) |
                EntryLocal(p, _) |
                EntryPat(p, _) |
                EntryBlock(p, _) |
                EntryStructCtor(p, _) |
                EntryLifetime(p, _) |
                EntryTyParam(p, _) |
                EntryVisibility(p, _) =>
                    id = p,

                EntryExpr(p, _) => {
                    last_expr = Some(id);
                    id = p;
                }

                RootCrate => {
                    return DepNode::Hir(DefId::local(CRATE_DEF_INDEX));
                }

                NotPresent =>
                    // Some nodes, notably macro definitions, are not
                    // present in the map for whatever reason, but
                    // they *do* have def-ids. So if we encounter an
                    // empty hole, check for that case.
                    return self.opt_local_def_id(id)
                               .map(|def_id| DepNode::Hir(def_id))
                               .unwrap_or_else(|| {
                                   bug!("Walking parents from `{}` \
                                         led to `NotPresent` at `{}`",
                                        id0, id)
                               }),
            }
        }
    }

    pub fn num_local_def_ids(&self) -> usize {
        self.definitions.len()
    }

    pub fn definitions(&self) -> &Definitions {
        &self.definitions
    }

    pub fn def_key(&self, def_id: DefId) -> DefKey {
        assert!(def_id.is_local());
        self.definitions.def_key(def_id.index)
    }

    pub fn def_path_from_id(&self, id: NodeId) -> Option<DefPath> {
        self.opt_local_def_id(id).map(|def_id| {
            self.def_path(def_id)
        })
    }

    pub fn def_path(&self, def_id: DefId) -> DefPath {
        assert!(def_id.is_local());
        self.definitions.def_path(def_id.index)
    }

    pub fn def_index_for_def_key(&self, def_key: DefKey) -> Option<DefIndex> {
        self.definitions.def_index_for_def_key(def_key)
    }

    pub fn local_def_id(&self, node: NodeId) -> DefId {
        self.opt_local_def_id(node).unwrap_or_else(|| {
            bug!("local_def_id: no entry for `{}`, which has a map of `{:?}`",
                 node, self.find_entry(node))
        })
    }

    pub fn opt_local_def_id(&self, node: NodeId) -> Option<DefId> {
        self.definitions.opt_local_def_id(node)
    }

    pub fn as_local_node_id(&self, def_id: DefId) -> Option<NodeId> {
        self.definitions.as_local_node_id(def_id)
    }

    fn entry_count(&self) -> usize {
        self.map.len()
    }

    fn find_entry(&self, id: NodeId) -> Option<MapEntry<'ast>> {
        self.map.get(id.as_usize()).cloned()
    }

    pub fn krate(&self) -> &'ast Crate {
        self.forest.krate()
    }

    pub fn trait_item(&self, id: TraitItemId) -> &'ast TraitItem {
        self.read(id.node_id);

        // NB: intentionally bypass `self.forest.krate()` so that we
        // do not trigger a read of the whole krate here
        self.forest.krate.trait_item(id)
    }

    pub fn impl_item(&self, id: ImplItemId) -> &'ast ImplItem {
        self.read(id.node_id);

        // NB: intentionally bypass `self.forest.krate()` so that we
        // do not trigger a read of the whole krate here
        self.forest.krate.impl_item(id)
    }

    pub fn body(&self, id: BodyId) -> &'ast Body {
        self.read(id.node_id);

        // NB: intentionally bypass `self.forest.krate()` so that we
        // do not trigger a read of the whole krate here
        self.forest.krate.body(id)
    }

    /// Returns the `NodeId` that corresponds to the definition of
    /// which this is the body of, i.e. a `fn`, `const` or `static`
    /// item (possibly associated), or a closure, or the body itself
    /// for embedded constant expressions (e.g. `N` in `[T; N]`).
    pub fn body_owner(&self, BodyId { node_id }: BodyId) -> NodeId {
        let parent = self.get_parent_node(node_id);
        if self.map[parent.as_usize()].is_body_owner(node_id) {
            parent
        } else {
            node_id
        }
    }

    pub fn body_owner_def_id(&self, id: BodyId) -> DefId {
        self.local_def_id(self.body_owner(id))
    }

    /// Get the attributes on the krate. This is preferable to
    /// invoking `krate.attrs` because it registers a tighter
    /// dep-graph access.
    pub fn krate_attrs(&self) -> &'ast [ast::Attribute] {
        let crate_root_def_id = DefId::local(CRATE_DEF_INDEX);
        self.dep_graph.read(DepNode::Hir(crate_root_def_id));
        &self.forest.krate.attrs
    }

    /// Retrieve the Node corresponding to `id`, panicking if it cannot
    /// be found.
    pub fn get(&self, id: NodeId) -> Node<'ast> {
        match self.find(id) {
            Some(node) => node, // read recorded by `find`
            None => bug!("couldn't find node id {} in the AST map", id)
        }
    }

    pub fn get_if_local(&self, id: DefId) -> Option<Node<'ast>> {
        self.as_local_node_id(id).map(|id| self.get(id)) // read recorded by `get`
    }

    /// Retrieve the Node corresponding to `id`, returning None if
    /// cannot be found.
    pub fn find(&self, id: NodeId) -> Option<Node<'ast>> {
        let result = self.find_entry(id).and_then(|x| x.to_node());
        if result.is_some() {
            self.read(id);
        }
        result
    }

    /// Similar to get_parent, returns the parent node id or id if there is no
    /// parent.
    /// This function returns the immediate parent in the AST, whereas get_parent
    /// returns the enclosing item. Note that this might not be the actual parent
    /// node in the AST - some kinds of nodes are not in the map and these will
    /// never appear as the parent_node. So you can always walk the parent_nodes
    /// from a node to the root of the ast (unless you get the same id back here
    /// that can happen if the id is not in the map itself or is just weird).
    pub fn get_parent_node(&self, id: NodeId) -> NodeId {
        self.find_entry(id).and_then(|x| x.parent_node()).unwrap_or(id)
    }

    /// Check if the node is an argument. An argument is a local variable whose
    /// immediate parent is an item or a closure.
    pub fn is_argument(&self, id: NodeId) -> bool {
        match self.find(id) {
            Some(NodeLocal(_)) => (),
            _ => return false,
        }
        match self.find(self.get_parent_node(id)) {
            Some(NodeItem(_)) |
            Some(NodeTraitItem(_)) |
            Some(NodeImplItem(_)) => true,
            Some(NodeExpr(e)) => {
                match e.node {
                    ExprClosure(..) => true,
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// If there is some error when walking the parents (e.g., a node does not
    /// have a parent in the map or a node can't be found), then we return the
    /// last good node id we found. Note that reaching the crate root (id == 0),
    /// is not an error, since items in the crate module have the crate root as
    /// parent.
    fn walk_parent_nodes<F>(&self, start_id: NodeId, found: F) -> Result<NodeId, NodeId>
        where F: Fn(&Node<'ast>) -> bool
    {
        let mut id = start_id;
        loop {
            let parent_node = self.get_parent_node(id);
            if parent_node == CRATE_NODE_ID {
                return Ok(CRATE_NODE_ID);
            }
            if parent_node == id {
                return Err(id);
            }

            let node = self.find_entry(parent_node);
            if node.is_none() {
                return Err(id);
            }
            let node = node.unwrap().to_node();
            match node {
                Some(ref node) => {
                    if found(node) {
                        return Ok(parent_node);
                    }
                }
                None => {
                    return Err(parent_node);
                }
            }
            id = parent_node;
        }
    }

    /// Retrieve the NodeId for `id`'s parent item, or `id` itself if no
    /// parent item is in this map. The "parent item" is the closest parent node
    /// in the AST which is recorded by the map and is an item, either an item
    /// in a module, trait, or impl.
    pub fn get_parent(&self, id: NodeId) -> NodeId {
        match self.walk_parent_nodes(id, |node| match *node {
            NodeItem(_) |
            NodeForeignItem(_) |
            NodeTraitItem(_) |
            NodeImplItem(_) => true,
            _ => false,
        }) {
            Ok(id) => id,
            Err(id) => id,
        }
    }

    /// Returns the NodeId of `id`'s nearest module parent, or `id` itself if no
    /// module parent is in this map.
    pub fn get_module_parent(&self, id: NodeId) -> NodeId {
        match self.walk_parent_nodes(id, |node| match *node {
            NodeItem(&Item { node: Item_::ItemMod(_), .. }) => true,
            _ => false,
        }) {
            Ok(id) => id,
            Err(id) => id,
        }
    }

    /// Returns the nearest enclosing scope. A scope is an item or block.
    /// FIXME it is not clear to me that all items qualify as scopes - statics
    /// and associated types probably shouldn't, for example. Behaviour in this
    /// regard should be expected to be highly unstable.
    pub fn get_enclosing_scope(&self, id: NodeId) -> Option<NodeId> {
        match self.walk_parent_nodes(id, |node| match *node {
            NodeItem(_) |
            NodeForeignItem(_) |
            NodeTraitItem(_) |
            NodeImplItem(_) |
            NodeBlock(_) => true,
            _ => false,
        }) {
            Ok(id) => Some(id),
            Err(_) => None,
        }
    }

    pub fn get_parent_did(&self, id: NodeId) -> DefId {
        self.local_def_id(self.get_parent(id))
    }

    pub fn get_foreign_abi(&self, id: NodeId) -> Abi {
        let parent = self.get_parent(id);
        let abi = match self.find_entry(parent) {
            Some(EntryItem(_, i)) => {
                match i.node {
                    ItemForeignMod(ref nm) => Some(nm.abi),
                    _ => None
                }
            }
            _ => None
        };
        match abi {
            Some(abi) => {
                self.read(id); // reveals some of the content of a node
                abi
            }
            None => bug!("expected foreign mod or inlined parent, found {}",
                          self.node_to_string(parent))
        }
    }

    pub fn expect_item(&self, id: NodeId) -> &'ast Item {
        match self.find(id) { // read recorded by `find`
            Some(NodeItem(item)) => item,
            _ => bug!("expected item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_impl_item(&self, id: NodeId) -> &'ast ImplItem {
        match self.find(id) {
            Some(NodeImplItem(item)) => item,
            _ => bug!("expected impl item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_trait_item(&self, id: NodeId) -> &'ast TraitItem {
        match self.find(id) {
            Some(NodeTraitItem(item)) => item,
            _ => bug!("expected trait item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_variant_data(&self, id: NodeId) -> &'ast VariantData {
        match self.find(id) {
            Some(NodeItem(i)) => {
                match i.node {
                    ItemStruct(ref struct_def, _) |
                    ItemUnion(ref struct_def, _) => struct_def,
                    _ => {
                        bug!("struct ID bound to non-struct {}",
                             self.node_to_string(id));
                    }
                }
            }
            Some(NodeStructCtor(data)) => data,
            Some(NodeVariant(variant)) => &variant.node.data,
            _ => {
                bug!("expected struct or variant, found {}",
                     self.node_to_string(id));
            }
        }
    }

    pub fn expect_variant(&self, id: NodeId) -> &'ast Variant {
        match self.find(id) {
            Some(NodeVariant(variant)) => variant,
            _ => bug!("expected variant, found {}", self.node_to_string(id)),
        }
    }

    pub fn expect_foreign_item(&self, id: NodeId) -> &'ast ForeignItem {
        match self.find(id) {
            Some(NodeForeignItem(item)) => item,
            _ => bug!("expected foreign item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_expr(&self, id: NodeId) -> &'ast Expr {
        match self.find(id) { // read recorded by find
            Some(NodeExpr(expr)) => expr,
            _ => bug!("expected expr, found {}", self.node_to_string(id))
        }
    }

    pub fn get_inlined_body(&self, def_id: DefId) -> Option<&'ast Body> {
        self.inlined_bodies.borrow().get(&def_id).map(|&body| {
            self.dep_graph.read(DepNode::MetaData(def_id));
            body
        })
    }

    pub fn intern_inlined_body(&self, def_id: DefId, body: Body) -> &'ast Body {
        let body = self.forest.inlined_bodies.alloc(body);
        self.inlined_bodies.borrow_mut().insert(def_id, body);
        body
    }

    /// Returns the name associated with the given NodeId's AST.
    pub fn name(&self, id: NodeId) -> Name {
        match self.get(id) {
            NodeItem(i) => i.name,
            NodeForeignItem(i) => i.name,
            NodeImplItem(ii) => ii.name,
            NodeTraitItem(ti) => ti.name,
            NodeVariant(v) => v.node.name,
            NodeField(f) => f.name,
            NodeLifetime(lt) => lt.name,
            NodeTyParam(tp) => tp.name,
            NodeLocal(&Pat { node: PatKind::Binding(_,_,l,_), .. }) => l.node,
            NodeStructCtor(_) => self.name(self.get_parent(id)),
            _ => bug!("no name for {}", self.node_to_string(id))
        }
    }

    /// Given a node ID, get a list of attributes associated with the AST
    /// corresponding to the Node ID
    pub fn attrs(&self, id: NodeId) -> &'ast [ast::Attribute] {
        self.read(id); // reveals attributes on the node
        let attrs = match self.find(id) {
            Some(NodeItem(i)) => Some(&i.attrs[..]),
            Some(NodeForeignItem(fi)) => Some(&fi.attrs[..]),
            Some(NodeTraitItem(ref ti)) => Some(&ti.attrs[..]),
            Some(NodeImplItem(ref ii)) => Some(&ii.attrs[..]),
            Some(NodeVariant(ref v)) => Some(&v.node.attrs[..]),
            Some(NodeField(ref f)) => Some(&f.attrs[..]),
            Some(NodeExpr(ref e)) => Some(&*e.attrs),
            Some(NodeStmt(ref s)) => Some(s.node.attrs()),
            // unit/tuple structs take the attributes straight from
            // the struct definition.
            Some(NodeStructCtor(_)) => {
                return self.attrs(self.get_parent(id));
            }
            _ => None
        };
        attrs.unwrap_or(&[])
    }

    /// Returns an iterator that yields the node id's with paths that
    /// match `parts`.  (Requires `parts` is non-empty.)
    ///
    /// For example, if given `parts` equal to `["bar", "quux"]`, then
    /// the iterator will produce node id's for items with paths
    /// such as `foo::bar::quux`, `bar::quux`, `other::bar::quux`, and
    /// any other such items it can find in the map.
    pub fn nodes_matching_suffix<'a>(&'a self, parts: &'a [String])
                                 -> NodesMatchingSuffix<'a, 'ast> {
        NodesMatchingSuffix {
            map: self,
            item_name: parts.last().unwrap(),
            in_which: &parts[..parts.len() - 1],
            idx: CRATE_NODE_ID,
        }
    }

    pub fn span(&self, id: NodeId) -> Span {
        self.read(id); // reveals span from node
        match self.find_entry(id) {
            Some(EntryItem(_, item)) => item.span,
            Some(EntryForeignItem(_, foreign_item)) => foreign_item.span,
            Some(EntryTraitItem(_, trait_method)) => trait_method.span,
            Some(EntryImplItem(_, impl_item)) => impl_item.span,
            Some(EntryVariant(_, variant)) => variant.span,
            Some(EntryField(_, field)) => field.span,
            Some(EntryExpr(_, expr)) => expr.span,
            Some(EntryStmt(_, stmt)) => stmt.span,
            Some(EntryTy(_, ty)) => ty.span,
            Some(EntryTraitRef(_, tr)) => tr.path.span,
            Some(EntryLocal(_, pat)) => pat.span,
            Some(EntryPat(_, pat)) => pat.span,
            Some(EntryBlock(_, block)) => block.span,
            Some(EntryStructCtor(_, _)) => self.expect_item(self.get_parent(id)).span,
            Some(EntryLifetime(_, lifetime)) => lifetime.span,
            Some(EntryTyParam(_, ty_param)) => ty_param.span,
            Some(EntryVisibility(_, &Visibility::Restricted { ref path, .. })) => path.span,
            Some(EntryVisibility(_, v)) => bug!("unexpected Visibility {:?}", v),

            Some(RootCrate) => self.forest.krate.span,
            Some(NotPresent) | None => {
                bug!("hir::map::Map::span: id not in map: {:?}", id)
            }
        }
    }

    pub fn span_if_local(&self, id: DefId) -> Option<Span> {
        self.as_local_node_id(id).map(|id| self.span(id))
    }

    pub fn node_to_string(&self, id: NodeId) -> String {
        node_id_to_string(self, id, true)
    }

    pub fn node_to_user_string(&self, id: NodeId) -> String {
        node_id_to_string(self, id, false)
    }

    pub fn node_to_pretty_string(&self, id: NodeId) -> String {
        print::to_string(self, |s| s.print_node(self.get(id)))
    }
}

pub struct NodesMatchingSuffix<'a, 'ast:'a> {
    map: &'a Map<'ast>,
    item_name: &'a String,
    in_which: &'a [String],
    idx: NodeId,
}

impl<'a, 'ast> NodesMatchingSuffix<'a, 'ast> {
    /// Returns true only if some suffix of the module path for parent
    /// matches `self.in_which`.
    ///
    /// In other words: let `[x_0,x_1,...,x_k]` be `self.in_which`;
    /// returns true if parent's path ends with the suffix
    /// `x_0::x_1::...::x_k`.
    fn suffix_matches(&self, parent: NodeId) -> bool {
        let mut cursor = parent;
        for part in self.in_which.iter().rev() {
            let (mod_id, mod_name) = match find_first_mod_parent(self.map, cursor) {
                None => return false,
                Some((node_id, name)) => (node_id, name),
            };
            if mod_name != &**part {
                return false;
            }
            cursor = self.map.get_parent(mod_id);
        }
        return true;

        // Finds the first mod in parent chain for `id`, along with
        // that mod's name.
        //
        // If `id` itself is a mod named `m` with parent `p`, then
        // returns `Some(id, m, p)`.  If `id` has no mod in its parent
        // chain, then returns `None`.
        fn find_first_mod_parent<'a>(map: &'a Map, mut id: NodeId) -> Option<(NodeId, Name)> {
            loop {
                match map.find(id) {
                    None => return None,
                    Some(NodeItem(item)) if item_is_mod(&item) =>
                        return Some((id, item.name)),
                    _ => {}
                }
                let parent = map.get_parent(id);
                if parent == id { return None }
                id = parent;
            }

            fn item_is_mod(item: &Item) -> bool {
                match item.node {
                    ItemMod(_) => true,
                    _ => false,
                }
            }
        }
    }

    // We are looking at some node `n` with a given name and parent
    // id; do their names match what I am seeking?
    fn matches_names(&self, parent_of_n: NodeId, name: Name) -> bool {
        name == &**self.item_name && self.suffix_matches(parent_of_n)
    }
}

impl<'a, 'ast> Iterator for NodesMatchingSuffix<'a, 'ast> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        loop {
            let idx = self.idx;
            if idx.as_usize() >= self.map.entry_count() {
                return None;
            }
            self.idx = NodeId::from_u32(self.idx.as_u32() + 1);
            let name = match self.map.find_entry(idx) {
                Some(EntryItem(_, n))       => n.name(),
                Some(EntryForeignItem(_, n))=> n.name(),
                Some(EntryTraitItem(_, n))  => n.name(),
                Some(EntryImplItem(_, n))   => n.name(),
                Some(EntryVariant(_, n))    => n.name(),
                Some(EntryField(_, n))      => n.name(),
                _ => continue,
            };
            if self.matches_names(self.map.get_parent(idx), name) {
                return Some(idx)
            }
        }
    }
}

trait Named {
    fn name(&self) -> Name;
}

impl<T:Named> Named for Spanned<T> { fn name(&self) -> Name { self.node.name() } }

impl Named for Item { fn name(&self) -> Name { self.name } }
impl Named for ForeignItem { fn name(&self) -> Name { self.name } }
impl Named for Variant_ { fn name(&self) -> Name { self.name } }
impl Named for StructField { fn name(&self) -> Name { self.name } }
impl Named for TraitItem { fn name(&self) -> Name { self.name } }
impl Named for ImplItem { fn name(&self) -> Name { self.name } }

pub fn map_crate<'ast>(forest: &'ast mut Forest,
                       definitions: Definitions)
                       -> Map<'ast> {
    let mut collector = NodeCollector::root(&forest.krate);
    intravisit::walk_crate(&mut collector, &forest.krate);
    let map = collector.map;

    if log_enabled!(::log::DEBUG) {
        // This only makes sense for ordered stores; note the
        // enumerate to count the number of entries.
        let (entries_less_1, _) = map.iter().filter(|&x| {
            match *x {
                NotPresent => false,
                _ => true
            }
        }).enumerate().last().expect("AST map was empty after folding?");

        let entries = entries_less_1 + 1;
        let vector_length = map.len();
        debug!("The AST map has {} entries with a maximum of {}: occupancy {:.1}%",
              entries, vector_length, (entries as f64 / vector_length as f64) * 100.);
    }

    Map {
        forest: forest,
        dep_graph: forest.dep_graph.clone(),
        map: map,
        definitions: definitions,
        inlined_bodies: RefCell::new(DefIdMap()),
    }
}

/// Identical to the `PpAnn` implementation for `hir::Crate`,
/// except it avoids creating a dependency on the whole crate.
impl<'ast> print::PpAnn for Map<'ast> {
    fn nested(&self, state: &mut print::State, nested: print::Nested) -> io::Result<()> {
        match nested {
            Nested::Item(id) => state.print_item(self.expect_item(id.id)),
            Nested::TraitItem(id) => state.print_trait_item(self.trait_item(id)),
            Nested::ImplItem(id) => state.print_impl_item(self.impl_item(id)),
            Nested::Body(id) => state.print_expr(&self.body(id).value),
            Nested::BodyArgPat(id, i) => state.print_pat(&self.body(id).arguments[i].pat)
        }
    }
}

impl<'a> print::State<'a> {
    pub fn print_node(&mut self, node: Node) -> io::Result<()> {
        match node {
            NodeItem(a)        => self.print_item(&a),
            NodeForeignItem(a) => self.print_foreign_item(&a),
            NodeTraitItem(a)   => self.print_trait_item(a),
            NodeImplItem(a)    => self.print_impl_item(a),
            NodeVariant(a)     => self.print_variant(&a),
            NodeExpr(a)        => self.print_expr(&a),
            NodeStmt(a)        => self.print_stmt(&a),
            NodeTy(a)          => self.print_type(&a),
            NodeTraitRef(a)    => self.print_trait_ref(&a),
            NodeLocal(a)       |
            NodePat(a)         => self.print_pat(&a),
            NodeBlock(a)       => {
                use syntax::print::pprust::PrintState;

                // containing cbox, will be closed by print-block at }
                self.cbox(print::indent_unit)?;
                // head-ibox, will be closed by print-block after {
                self.ibox(0)?;
                self.print_block(&a)
            }
            NodeLifetime(a)    => self.print_lifetime(&a),
            NodeVisibility(a)  => self.print_visibility(&a),
            NodeTyParam(_)     => bug!("cannot print TyParam"),
            NodeField(_)       => bug!("cannot print StructField"),
            // these cases do not carry enough information in the
            // ast_map to reconstruct their full structure for pretty
            // printing.
            NodeStructCtor(_)  => bug!("cannot print isolated StructCtor"),
        }
    }
}

fn node_id_to_string(map: &Map, id: NodeId, include_id: bool) -> String {
    let id_str = format!(" (id={})", id);
    let id_str = if include_id { &id_str[..] } else { "" };

    let path_str = || {
        // This functionality is used for debugging, try to use TyCtxt to get
        // the user-friendly path, otherwise fall back to stringifying DefPath.
        ::ty::tls::with_opt(|tcx| {
            if let Some(tcx) = tcx {
                tcx.node_path_str(id)
            } else if let Some(path) = map.def_path_from_id(id) {
                path.data.into_iter().map(|elem| {
                    elem.data.to_string()
                }).collect::<Vec<_>>().join("::")
            } else {
                String::from("<missing path>")
            }
        })
    };

    match map.find(id) {
        Some(NodeItem(item)) => {
            let item_str = match item.node {
                ItemExternCrate(..) => "extern crate",
                ItemUse(..) => "use",
                ItemStatic(..) => "static",
                ItemConst(..) => "const",
                ItemFn(..) => "fn",
                ItemMod(..) => "mod",
                ItemForeignMod(..) => "foreign mod",
                ItemTy(..) => "ty",
                ItemEnum(..) => "enum",
                ItemStruct(..) => "struct",
                ItemUnion(..) => "union",
                ItemTrait(..) => "trait",
                ItemImpl(..) => "impl",
                ItemDefaultImpl(..) => "default impl",
            };
            format!("{} {}{}", item_str, path_str(), id_str)
        }
        Some(NodeForeignItem(_)) => {
            format!("foreign item {}{}", path_str(), id_str)
        }
        Some(NodeImplItem(ii)) => {
            match ii.node {
                ImplItemKind::Const(..) => {
                    format!("assoc const {} in {}{}", ii.name, path_str(), id_str)
                }
                ImplItemKind::Method(..) => {
                    format!("method {} in {}{}", ii.name, path_str(), id_str)
                }
                ImplItemKind::Type(_) => {
                    format!("assoc type {} in {}{}", ii.name, path_str(), id_str)
                }
            }
        }
        Some(NodeTraitItem(ti)) => {
            let kind = match ti.node {
                TraitItemKind::Const(..) => "assoc constant",
                TraitItemKind::Method(..) => "trait method",
                TraitItemKind::Type(..) => "assoc type",
            };

            format!("{} {} in {}{}", kind, ti.name, path_str(), id_str)
        }
        Some(NodeVariant(ref variant)) => {
            format!("variant {} in {}{}",
                    variant.node.name,
                    path_str(), id_str)
        }
        Some(NodeField(ref field)) => {
            format!("field {} in {}{}",
                    field.name,
                    path_str(), id_str)
        }
        Some(NodeExpr(_)) => {
            format!("expr {}{}", map.node_to_pretty_string(id), id_str)
        }
        Some(NodeStmt(_)) => {
            format!("stmt {}{}", map.node_to_pretty_string(id), id_str)
        }
        Some(NodeTy(_)) => {
            format!("type {}{}", map.node_to_pretty_string(id), id_str)
        }
        Some(NodeTraitRef(_)) => {
            format!("trait_ref {}{}", map.node_to_pretty_string(id), id_str)
        }
        Some(NodeLocal(_)) => {
            format!("local {}{}", map.node_to_pretty_string(id), id_str)
        }
        Some(NodePat(_)) => {
            format!("pat {}{}", map.node_to_pretty_string(id), id_str)
        }
        Some(NodeBlock(_)) => {
            format!("block {}{}", map.node_to_pretty_string(id), id_str)
        }
        Some(NodeStructCtor(_)) => {
            format!("struct_ctor {}{}", path_str(), id_str)
        }
        Some(NodeLifetime(_)) => {
            format!("lifetime {}{}", map.node_to_pretty_string(id), id_str)
        }
        Some(NodeTyParam(ref ty_param)) => {
            format!("typaram {:?}{}", ty_param, id_str)
        }
        Some(NodeVisibility(ref vis)) => {
            format!("visibility {:?}{}", vis, id_str)
        }
        None => {
            format!("unknown node{}", id_str)
        }
    }
}
