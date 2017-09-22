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
                            DisambiguatedDefPathData, DefPathHash};

use dep_graph::{DepGraph, DepNode, DepKind, DepNodeIndex};

use hir::def_id::{CRATE_DEF_INDEX, DefId, DefIndexAddressSpace};

use syntax::abi::Abi;
use syntax::ast::{self, Name, NodeId, CRATE_NODE_ID};
use syntax::codemap::Spanned;
use syntax_pos::Span;

use hir::*;
use hir::print::Nested;
use util::nodemap::{DefIdMap, FxHashMap};

use arena::TypedArena;
use std::cell::RefCell;
use std::io;

pub mod blocks;
mod collector;
mod def_collector;
pub mod definitions;
mod hir_id_validator;

pub const ITEM_LIKE_SPACE: DefIndexAddressSpace = DefIndexAddressSpace::Low;
pub const REGULAR_SPACE: DefIndexAddressSpace = DefIndexAddressSpace::High;

#[derive(Copy, Clone, Debug)]
pub enum Node<'hir> {
    NodeItem(&'hir Item),
    NodeForeignItem(&'hir ForeignItem),
    NodeTraitItem(&'hir TraitItem),
    NodeImplItem(&'hir ImplItem),
    NodeVariant(&'hir Variant),
    NodeField(&'hir StructField),
    NodeExpr(&'hir Expr),
    NodeStmt(&'hir Stmt),
    NodeTy(&'hir Ty),
    NodeTraitRef(&'hir TraitRef),
    NodeBinding(&'hir Pat),
    NodePat(&'hir Pat),
    NodeBlock(&'hir Block),
    NodeLocal(&'hir Local),

    /// NodeStructCtor represents a tuple struct.
    NodeStructCtor(&'hir VariantData),

    NodeLifetime(&'hir Lifetime),
    NodeTyParam(&'hir TyParam),
    NodeVisibility(&'hir Visibility),
}

/// Represents an entry and its parent NodeID.
/// The odd layout is to bring down the total size.
#[derive(Copy, Debug)]
enum MapEntry<'hir> {
    /// Placeholder for holes in the map.
    NotPresent,

    /// All the node types, with a parent ID.
    EntryItem(NodeId, DepNodeIndex, &'hir Item),
    EntryForeignItem(NodeId, DepNodeIndex, &'hir ForeignItem),
    EntryTraitItem(NodeId, DepNodeIndex, &'hir TraitItem),
    EntryImplItem(NodeId, DepNodeIndex, &'hir ImplItem),
    EntryVariant(NodeId, DepNodeIndex, &'hir Variant),
    EntryField(NodeId, DepNodeIndex, &'hir StructField),
    EntryExpr(NodeId, DepNodeIndex, &'hir Expr),
    EntryStmt(NodeId, DepNodeIndex, &'hir Stmt),
    EntryTy(NodeId, DepNodeIndex, &'hir Ty),
    EntryTraitRef(NodeId, DepNodeIndex, &'hir TraitRef),
    EntryBinding(NodeId, DepNodeIndex, &'hir Pat),
    EntryPat(NodeId, DepNodeIndex, &'hir Pat),
    EntryBlock(NodeId, DepNodeIndex, &'hir Block),
    EntryStructCtor(NodeId, DepNodeIndex, &'hir VariantData),
    EntryLifetime(NodeId, DepNodeIndex, &'hir Lifetime),
    EntryTyParam(NodeId, DepNodeIndex, &'hir TyParam),
    EntryVisibility(NodeId, DepNodeIndex, &'hir Visibility),
    EntryLocal(NodeId, DepNodeIndex, &'hir Local),

    /// Roots for node trees. The DepNodeIndex is the dependency node of the
    /// crate's root module.
    RootCrate(DepNodeIndex),
}

impl<'hir> Clone for MapEntry<'hir> {
    fn clone(&self) -> MapEntry<'hir> {
        *self
    }
}

impl<'hir> MapEntry<'hir> {
    fn parent_node(self) -> Option<NodeId> {
        Some(match self {
            EntryItem(id, _, _) => id,
            EntryForeignItem(id, _, _) => id,
            EntryTraitItem(id, _, _) => id,
            EntryImplItem(id, _, _) => id,
            EntryVariant(id, _, _) => id,
            EntryField(id, _, _) => id,
            EntryExpr(id, _, _) => id,
            EntryStmt(id, _, _) => id,
            EntryTy(id, _, _) => id,
            EntryTraitRef(id, _, _) => id,
            EntryBinding(id, _, _) => id,
            EntryPat(id, _, _) => id,
            EntryBlock(id, _, _) => id,
            EntryStructCtor(id, _, _) => id,
            EntryLifetime(id, _, _) => id,
            EntryTyParam(id, _, _) => id,
            EntryVisibility(id, _, _) => id,
            EntryLocal(id, _, _) => id,

            NotPresent |
            RootCrate(_) => return None,
        })
    }

    fn to_node(self) -> Option<Node<'hir>> {
        Some(match self {
            EntryItem(_, _, n) => NodeItem(n),
            EntryForeignItem(_, _, n) => NodeForeignItem(n),
            EntryTraitItem(_, _, n) => NodeTraitItem(n),
            EntryImplItem(_, _, n) => NodeImplItem(n),
            EntryVariant(_, _, n) => NodeVariant(n),
            EntryField(_, _, n) => NodeField(n),
            EntryExpr(_, _, n) => NodeExpr(n),
            EntryStmt(_, _, n) => NodeStmt(n),
            EntryTy(_, _, n) => NodeTy(n),
            EntryTraitRef(_, _, n) => NodeTraitRef(n),
            EntryBinding(_, _, n) => NodeBinding(n),
            EntryPat(_, _, n) => NodePat(n),
            EntryBlock(_, _, n) => NodeBlock(n),
            EntryStructCtor(_, _, n) => NodeStructCtor(n),
            EntryLifetime(_, _, n) => NodeLifetime(n),
            EntryTyParam(_, _, n) => NodeTyParam(n),
            EntryVisibility(_, _, n) => NodeVisibility(n),
            EntryLocal(_, _, n) => NodeLocal(n),

            NotPresent |
            RootCrate(_) => return None
        })
    }

    fn associated_body(self) -> Option<BodyId> {
        match self {
            EntryItem(_, _, item) => {
                match item.node {
                    ItemConst(_, body) |
                    ItemStatic(.., body) |
                    ItemFn(_, _, _, _, _, body) => Some(body),
                    _ => None,
                }
            }

            EntryTraitItem(_, _, item) => {
                match item.node {
                    TraitItemKind::Const(_, Some(body)) |
                    TraitItemKind::Method(_, TraitMethod::Provided(body)) => Some(body),
                    _ => None
                }
            }

            EntryImplItem(_, _, item) => {
                match item.node {
                    ImplItemKind::Const(_, body) |
                    ImplItemKind::Method(_, body) => Some(body),
                    _ => None,
                }
            }

            EntryExpr(_, _, expr) => {
                match expr.node {
                    ExprClosure(.., body, _, _) => Some(body),
                    _ => None,
                }
            }

            _ => None
        }
    }

    fn is_body_owner(self, node_id: NodeId) -> bool {
        match self.associated_body() {
            Some(b) => b.node_id == node_id,
            None => false,
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
            krate,
            dep_graph: dep_graph.clone(),
            inlined_bodies: TypedArena::new()
        }
    }

    pub fn krate<'hir>(&'hir self) -> &'hir Crate {
        self.dep_graph.read(DepNode::new_no_params(DepKind::Krate));
        &self.krate
    }
}

/// Represents a mapping from Node IDs to AST elements and their parent
/// Node IDs
#[derive(Clone)]
pub struct Map<'hir> {
    /// The backing storage for all the AST nodes.
    pub forest: &'hir Forest,

    /// Same as the dep_graph in forest, just available with one fewer
    /// deref. This is a gratuitous micro-optimization.
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
    map: Vec<MapEntry<'hir>>,

    definitions: &'hir Definitions,

    /// Bodies inlined from other crates are cached here.
    inlined_bodies: RefCell<DefIdMap<&'hir Body>>,

    /// The reverse mapping of `node_to_hir_id`.
    hir_to_node_id: FxHashMap<HirId, NodeId>,
}

impl<'hir> Map<'hir> {
    /// Registers a read in the dependency graph of the AST node with
    /// the given `id`. This needs to be called each time a public
    /// function returns the HIR for a node -- in other words, when it
    /// "reveals" the content of a node to the caller (who might not
    /// otherwise have had access to those contents, and hence needs a
    /// read recorded). If the function just returns a DefId or
    /// NodeId, no actual content was returned, so no read is needed.
    pub fn read(&self, id: NodeId) {
        let entry = self.map[id.as_usize()];
        match entry {
            EntryItem(_, dep_node_index, _) |
            EntryTraitItem(_, dep_node_index, _) |
            EntryImplItem(_, dep_node_index, _) |
            EntryVariant(_, dep_node_index, _) |
            EntryForeignItem(_, dep_node_index, _) |
            EntryField(_, dep_node_index, _) |
            EntryStmt(_, dep_node_index, _) |
            EntryTy(_, dep_node_index, _) |
            EntryTraitRef(_, dep_node_index, _) |
            EntryBinding(_, dep_node_index, _) |
            EntryPat(_, dep_node_index, _) |
            EntryBlock(_, dep_node_index, _) |
            EntryStructCtor(_, dep_node_index, _) |
            EntryLifetime(_, dep_node_index, _) |
            EntryTyParam(_, dep_node_index, _) |
            EntryVisibility(_, dep_node_index, _) |
            EntryExpr(_, dep_node_index, _) |
            EntryLocal(_, dep_node_index, _) |
            RootCrate(dep_node_index) => {
                self.dep_graph.read_index(dep_node_index);
            }
            NotPresent => {
                // Some nodes, notably macro definitions, are not
                // present in the map for whatever reason, but
                // they *do* have def-ids. So if we encounter an
                // empty hole, check for that case.
                if let Some(def_index) = self.definitions.opt_def_index(id) {
                    let def_path_hash = self.definitions.def_path_hash(def_index);
                    self.dep_graph.read(def_path_hash.to_dep_node(DepKind::Hir));
                } else {
                    bug!("called HirMap::read() with invalid NodeId")
                }
            }
        }
    }

    #[inline]
    pub fn definitions(&self) -> &'hir Definitions {
        self.definitions
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

    #[inline]
    pub fn local_def_id(&self, node: NodeId) -> DefId {
        self.opt_local_def_id(node).unwrap_or_else(|| {
            bug!("local_def_id: no entry for `{}`, which has a map of `{:?}`",
                 node, self.find_entry(node))
        })
    }

    #[inline]
    pub fn opt_local_def_id(&self, node: NodeId) -> Option<DefId> {
        self.definitions.opt_local_def_id(node)
    }

    #[inline]
    pub fn as_local_node_id(&self, def_id: DefId) -> Option<NodeId> {
        self.definitions.as_local_node_id(def_id)
    }

    #[inline]
    pub fn hir_to_node_id(&self, hir_id: HirId) -> NodeId {
        self.hir_to_node_id[&hir_id]
    }

    #[inline]
    pub fn node_to_hir_id(&self, node_id: NodeId) -> HirId {
        self.definitions.node_to_hir_id(node_id)
    }

    #[inline]
    pub fn def_index_to_hir_id(&self, def_index: DefIndex) -> HirId {
        self.definitions.def_index_to_hir_id(def_index)
    }

    #[inline]
    pub fn def_index_to_node_id(&self, def_index: DefIndex) -> NodeId {
        self.definitions.as_local_node_id(DefId::local(def_index)).unwrap()
    }

    fn entry_count(&self) -> usize {
        self.map.len()
    }

    fn find_entry(&self, id: NodeId) -> Option<MapEntry<'hir>> {
        self.map.get(id.as_usize()).cloned()
    }

    pub fn krate(&self) -> &'hir Crate {
        self.forest.krate()
    }

    pub fn trait_item(&self, id: TraitItemId) -> &'hir TraitItem {
        self.read(id.node_id);

        // NB: intentionally bypass `self.forest.krate()` so that we
        // do not trigger a read of the whole krate here
        self.forest.krate.trait_item(id)
    }

    pub fn impl_item(&self, id: ImplItemId) -> &'hir ImplItem {
        self.read(id.node_id);

        // NB: intentionally bypass `self.forest.krate()` so that we
        // do not trigger a read of the whole krate here
        self.forest.krate.impl_item(id)
    }

    pub fn body(&self, id: BodyId) -> &'hir Body {
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

    /// Given a node id, returns the `BodyId` associated with it,
    /// if the node is a body owner, otherwise returns `None`.
    pub fn maybe_body_owned_by(&self, id: NodeId) -> Option<BodyId> {
        if let Some(entry) = self.find_entry(id) {
            if let Some(body_id) = entry.associated_body() {
                // For item-like things and closures, the associated
                // body has its own distinct id, and that is returned
                // by `associated_body`.
                Some(body_id)
            } else {
                // For some expressions, the expression is its own body.
                if let EntryExpr(_, _, expr) = entry {
                    Some(BodyId { node_id: expr.id })
                } else {
                    None
                }
            }
        } else {
            bug!("no entry for id `{}`", id)
        }
    }

    /// Given a body owner's id, returns the `BodyId` associated with it.
    pub fn body_owned_by(&self, id: NodeId) -> BodyId {
        self.maybe_body_owned_by(id).unwrap_or_else(|| {
            span_bug!(self.span(id), "body_owned_by: {} has no associated body",
                      self.node_to_string(id));
        })
    }

    pub fn ty_param_owner(&self, id: NodeId) -> NodeId {
        match self.get(id) {
            NodeItem(&Item { node: ItemTrait(..), .. }) => id,
            NodeTyParam(_) => self.get_parent_node(id),
            _ => {
                bug!("ty_param_owner: {} not a type parameter",
                    self.node_to_string(id))
            }
        }
    }

    pub fn ty_param_name(&self, id: NodeId) -> Name {
        match self.get(id) {
            NodeItem(&Item { node: ItemTrait(..), .. }) => {
                keywords::SelfType.name()
            }
            NodeTyParam(tp) => tp.name,
            _ => {
                bug!("ty_param_name: {} not a type parameter",
                    self.node_to_string(id))
            }
        }
    }

    pub fn trait_impls(&self, trait_did: DefId) -> &'hir [NodeId] {
        self.dep_graph.read(DepNode::new_no_params(DepKind::AllLocalTraitImpls));

        // NB: intentionally bypass `self.forest.krate()` so that we
        // do not trigger a read of the whole krate here
        self.forest.krate.trait_impls.get(&trait_did).map_or(&[], |xs| &xs[..])
    }

    pub fn trait_default_impl(&self, trait_did: DefId) -> Option<NodeId> {
        self.dep_graph.read(DepNode::new_no_params(DepKind::AllLocalTraitImpls));

        // NB: intentionally bypass `self.forest.krate()` so that we
        // do not trigger a read of the whole krate here
        self.forest.krate.trait_default_impl.get(&trait_did).cloned()
    }

    pub fn trait_is_auto(&self, trait_did: DefId) -> bool {
        self.trait_default_impl(trait_did).is_some()
    }

    /// Get the attributes on the krate. This is preferable to
    /// invoking `krate.attrs` because it registers a tighter
    /// dep-graph access.
    pub fn krate_attrs(&self) -> &'hir [ast::Attribute] {
        let def_path_hash = self.definitions.def_path_hash(CRATE_DEF_INDEX);

        self.dep_graph.read(def_path_hash.to_dep_node(DepKind::Hir));
        &self.forest.krate.attrs
    }

    /// Retrieve the Node corresponding to `id`, panicking if it cannot
    /// be found.
    pub fn get(&self, id: NodeId) -> Node<'hir> {
        match self.find(id) {
            Some(node) => node, // read recorded by `find`
            None => bug!("couldn't find node id {} in the AST map", id)
        }
    }

    pub fn get_if_local(&self, id: DefId) -> Option<Node<'hir>> {
        self.as_local_node_id(id).map(|id| self.get(id)) // read recorded by `get`
    }

    /// Retrieve the Node corresponding to `id`, returning None if
    /// cannot be found.
    pub fn find(&self, id: NodeId) -> Option<Node<'hir>> {
        let result = self.find_entry(id).and_then(|x| x.to_node());
        if result.is_some() {
            self.read(id);
        }
        result
    }

    /// Similar to get_parent, returns the parent node id or id if there is no
    /// parent. Note that the parent may be CRATE_NODE_ID, which is not itself
    /// present in the map -- so passing the return value of get_parent_node to
    /// get may actually panic.
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
            Some(NodeBinding(_)) => (),
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
    fn walk_parent_nodes<F, F2>(&self,
                                start_id: NodeId,
                                found: F,
                                bail_early: F2)
        -> Result<NodeId, NodeId>
        where F: Fn(&Node<'hir>) -> bool, F2: Fn(&Node<'hir>) -> bool
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
                    } else if bail_early(node) {
                        return Err(parent_node);
                    }
                }
                None => {
                    return Err(parent_node);
                }
            }
            id = parent_node;
        }
    }

    /// Retrieve the NodeId for `id`'s enclosing method, unless there's a
    /// `while` or `loop` before reaching it, as block tail returns are not
    /// available in them.
    ///
    /// ```
    /// fn foo(x: usize) -> bool {
    ///     if x == 1 {
    ///         true  // `get_return_block` gets passed the `id` corresponding
    ///     } else {  // to this, it will return `foo`'s `NodeId`.
    ///         false
    ///     }
    /// }
    /// ```
    ///
    /// ```
    /// fn foo(x: usize) -> bool {
    ///     loop {
    ///         true  // `get_return_block` gets passed the `id` corresponding
    ///     }         // to this, it will return `None`.
    ///     false
    /// }
    /// ```
    pub fn get_return_block(&self, id: NodeId) -> Option<NodeId> {
        let match_fn = |node: &Node| {
            match *node {
                NodeItem(_) |
                NodeForeignItem(_) |
                NodeTraitItem(_) |
                NodeImplItem(_) => true,
                _ => false,
            }
        };
        let match_non_returning_block = |node: &Node| {
            match *node {
                NodeExpr(ref expr) => {
                    match expr.node {
                        ExprWhile(..) | ExprLoop(..) => true,
                        _ => false,
                    }
                }
                _ => false,
            }
        };

        match self.walk_parent_nodes(id, match_fn, match_non_returning_block) {
            Ok(id) => Some(id),
            Err(_) => None,
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
        }, |_| false) {
            Ok(id) => id,
            Err(id) => id,
        }
    }

    /// Returns the NodeId of `id`'s nearest module parent, or `id` itself if no
    /// module parent is in this map.
    pub fn get_module_parent(&self, id: NodeId) -> DefId {
        let id = match self.walk_parent_nodes(id, |node| match *node {
            NodeItem(&Item { node: Item_::ItemMod(_), .. }) => true,
            _ => false,
        }, |_| false) {
            Ok(id) => id,
            Err(id) => id,
        };
        self.local_def_id(id)
    }

    /// Returns the nearest enclosing scope. A scope is an item or block.
    /// FIXME it is not clear to me that all items qualify as scopes - statics
    /// and associated types probably shouldn't, for example. Behavior in this
    /// regard should be expected to be highly unstable.
    pub fn get_enclosing_scope(&self, id: NodeId) -> Option<NodeId> {
        match self.walk_parent_nodes(id, |node| match *node {
            NodeItem(_) |
            NodeForeignItem(_) |
            NodeTraitItem(_) |
            NodeImplItem(_) |
            NodeBlock(_) => true,
            _ => false,
        }, |_| false) {
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
            Some(EntryItem(_, _, i)) => {
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

    pub fn expect_item(&self, id: NodeId) -> &'hir Item {
        match self.find(id) { // read recorded by `find`
            Some(NodeItem(item)) => item,
            _ => bug!("expected item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_impl_item(&self, id: NodeId) -> &'hir ImplItem {
        match self.find(id) {
            Some(NodeImplItem(item)) => item,
            _ => bug!("expected impl item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_trait_item(&self, id: NodeId) -> &'hir TraitItem {
        match self.find(id) {
            Some(NodeTraitItem(item)) => item,
            _ => bug!("expected trait item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_variant_data(&self, id: NodeId) -> &'hir VariantData {
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

    pub fn expect_variant(&self, id: NodeId) -> &'hir Variant {
        match self.find(id) {
            Some(NodeVariant(variant)) => variant,
            _ => bug!("expected variant, found {}", self.node_to_string(id)),
        }
    }

    pub fn expect_foreign_item(&self, id: NodeId) -> &'hir ForeignItem {
        match self.find(id) {
            Some(NodeForeignItem(item)) => item,
            _ => bug!("expected foreign item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_expr(&self, id: NodeId) -> &'hir Expr {
        match self.find(id) { // read recorded by find
            Some(NodeExpr(expr)) => expr,
            _ => bug!("expected expr, found {}", self.node_to_string(id))
        }
    }

    pub fn get_inlined_body_untracked(&self, def_id: DefId) -> Option<&'hir Body> {
        self.inlined_bodies.borrow().get(&def_id).cloned()
    }

    pub fn intern_inlined_body(&self, def_id: DefId, body: Body) -> &'hir Body {
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
            NodeBinding(&Pat { node: PatKind::Binding(_,_,l,_), .. }) => l.node,
            NodeStructCtor(_) => self.name(self.get_parent(id)),
            _ => bug!("no name for {}", self.node_to_string(id))
        }
    }

    /// Given a node ID, get a list of attributes associated with the AST
    /// corresponding to the Node ID
    pub fn attrs(&self, id: NodeId) -> &'hir [ast::Attribute] {
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
                                 -> NodesMatchingSuffix<'a, 'hir> {
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
            Some(EntryItem(_, _, item)) => item.span,
            Some(EntryForeignItem(_, _, foreign_item)) => foreign_item.span,
            Some(EntryTraitItem(_, _, trait_method)) => trait_method.span,
            Some(EntryImplItem(_, _, impl_item)) => impl_item.span,
            Some(EntryVariant(_, _, variant)) => variant.span,
            Some(EntryField(_, _, field)) => field.span,
            Some(EntryExpr(_, _, expr)) => expr.span,
            Some(EntryStmt(_, _, stmt)) => stmt.span,
            Some(EntryTy(_, _, ty)) => ty.span,
            Some(EntryTraitRef(_, _, tr)) => tr.path.span,
            Some(EntryBinding(_, _, pat)) => pat.span,
            Some(EntryPat(_, _, pat)) => pat.span,
            Some(EntryBlock(_, _, block)) => block.span,
            Some(EntryStructCtor(_, _, _)) => self.expect_item(self.get_parent(id)).span,
            Some(EntryLifetime(_, _, lifetime)) => lifetime.span,
            Some(EntryTyParam(_, _, ty_param)) => ty_param.span,
            Some(EntryVisibility(_, _, &Visibility::Restricted { ref path, .. })) => path.span,
            Some(EntryVisibility(_, _, v)) => bug!("unexpected Visibility {:?}", v),
            Some(EntryLocal(_, _, local)) => local.span,

            Some(RootCrate(_)) => self.forest.krate.span,
            Some(NotPresent) | None => {
                // Some nodes, notably macro definitions, are not
                // present in the map for whatever reason, but
                // they *do* have def-ids. So if we encounter an
                // empty hole, check for that case.
                if let Some(def_index) = self.definitions.opt_def_index(id) {
                    let def_path_hash = self.definitions.def_path_hash(def_index);
                    self.dep_graph.read(def_path_hash.to_dep_node(DepKind::Hir));
                    DUMMY_SP
                } else {
                    bug!("hir::map::Map::span: id not in map: {:?}", id)
                }
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

pub struct NodesMatchingSuffix<'a, 'hir:'a> {
    map: &'a Map<'hir>,
    item_name: &'a String,
    in_which: &'a [String],
    idx: NodeId,
}

impl<'a, 'hir> NodesMatchingSuffix<'a, 'hir> {
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

impl<'a, 'hir> Iterator for NodesMatchingSuffix<'a, 'hir> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        loop {
            let idx = self.idx;
            if idx.as_usize() >= self.map.entry_count() {
                return None;
            }
            self.idx = NodeId::from_u32(self.idx.as_u32() + 1);
            let name = match self.map.find_entry(idx) {
                Some(EntryItem(_, _, n))       => n.name(),
                Some(EntryForeignItem(_, _, n))=> n.name(),
                Some(EntryTraitItem(_, _, n))  => n.name(),
                Some(EntryImplItem(_, _, n))   => n.name(),
                Some(EntryVariant(_, _, n))    => n.name(),
                Some(EntryField(_, _, n))      => n.name(),
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

pub fn map_crate<'hir>(forest: &'hir mut Forest,
                       definitions: &'hir Definitions)
                       -> Map<'hir> {
    let map = {
        let mut collector = NodeCollector::root(&forest.krate,
                                                &forest.dep_graph,
                                                &definitions);
        intravisit::walk_crate(&mut collector, &forest.krate);
        collector.into_map()
    };

    if log_enabled!(::log::LogLevel::Debug) {
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

    // Build the reverse mapping of `node_to_hir_id`.
    let hir_to_node_id = definitions.node_to_hir_id.iter_enumerated()
        .map(|(node_id, &hir_id)| (hir_id, node_id)).collect();

    let map = Map {
        forest,
        dep_graph: forest.dep_graph.clone(),
        map,
        hir_to_node_id,
        definitions,
        inlined_bodies: RefCell::new(DefIdMap()),
    };

    hir_id_validator::check_crate(&map);

    map
}

/// Identical to the `PpAnn` implementation for `hir::Crate`,
/// except it avoids creating a dependency on the whole crate.
impl<'hir> print::PpAnn for Map<'hir> {
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
            NodeBinding(a)       |
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
            // hir_map to reconstruct their full structure for pretty
            // printing.
            NodeStructCtor(_)  => bug!("cannot print isolated StructCtor"),
            NodeLocal(a)       => self.print_local_decl(&a),
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
                ItemGlobalAsm(..) => "global asm",
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
        Some(NodeBinding(_)) => {
            format!("local {}{}", map.node_to_pretty_string(id), id_str)
        }
        Some(NodePat(_)) => {
            format!("pat {}{}", map.node_to_pretty_string(id), id_str)
        }
        Some(NodeBlock(_)) => {
            format!("block {}{}", map.node_to_pretty_string(id), id_str)
        }
        Some(NodeLocal(_)) => {
            format!("local {}{}", map.node_to_pretty_string(id), id_str)
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
