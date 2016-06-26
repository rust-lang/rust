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
use self::def_collector::DefCollector;
pub use self::definitions::{Definitions, DefKey, DefPath, DefPathData,
                            DisambiguatedDefPathData, InlinedRootPath};

use dep_graph::{DepGraph, DepNode};

use middle::cstore::InlinedItem;
use middle::cstore::InlinedItem as II;
use hir::def_id::{CRATE_DEF_INDEX, DefId, DefIndex};

use syntax::abi::Abi;
use syntax::ast::{self, Name, NodeId, DUMMY_NODE_ID, };
use syntax::codemap::Spanned;
use syntax::visit;
use syntax_pos::Span;

use hir::*;
use hir::fold::Folder;
use hir::print as pprust;

use arena::TypedArena;
use std::cell::RefCell;
use std::io;
use std::mem;

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
    NodeExpr(&'ast Expr),
    NodeStmt(&'ast Stmt),
    NodeLocal(&'ast Pat),
    NodePat(&'ast Pat),
    NodeBlock(&'ast Block),

    /// NodeStructCtor represents a tuple struct.
    NodeStructCtor(&'ast VariantData),

    NodeLifetime(&'ast Lifetime),
    NodeTyParam(&'ast TyParam)
}

/// Represents an entry and its parent NodeID.
/// The odd layout is to bring down the total size.
#[derive(Copy, Debug)]
pub enum MapEntry<'ast> {
    /// Placeholder for holes in the map.
    NotPresent,

    /// All the node types, with a parent ID.
    EntryItem(NodeId, &'ast Item),
    EntryForeignItem(NodeId, &'ast ForeignItem),
    EntryTraitItem(NodeId, &'ast TraitItem),
    EntryImplItem(NodeId, &'ast ImplItem),
    EntryVariant(NodeId, &'ast Variant),
    EntryExpr(NodeId, &'ast Expr),
    EntryStmt(NodeId, &'ast Stmt),
    EntryLocal(NodeId, &'ast Pat),
    EntryPat(NodeId, &'ast Pat),
    EntryBlock(NodeId, &'ast Block),
    EntryStructCtor(NodeId, &'ast VariantData),
    EntryLifetime(NodeId, &'ast Lifetime),
    EntryTyParam(NodeId, &'ast TyParam),

    /// Roots for node trees.
    RootCrate,
    RootInlinedParent(&'ast InlinedItem)
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
            NodeExpr(n) => EntryExpr(p, n),
            NodeStmt(n) => EntryStmt(p, n),
            NodeLocal(n) => EntryLocal(p, n),
            NodePat(n) => EntryPat(p, n),
            NodeBlock(n) => EntryBlock(p, n),
            NodeStructCtor(n) => EntryStructCtor(p, n),
            NodeLifetime(n) => EntryLifetime(p, n),
            NodeTyParam(n) => EntryTyParam(p, n),
        }
    }

    fn parent_node(self) -> Option<NodeId> {
        Some(match self {
            EntryItem(id, _) => id,
            EntryForeignItem(id, _) => id,
            EntryTraitItem(id, _) => id,
            EntryImplItem(id, _) => id,
            EntryVariant(id, _) => id,
            EntryExpr(id, _) => id,
            EntryStmt(id, _) => id,
            EntryLocal(id, _) => id,
            EntryPat(id, _) => id,
            EntryBlock(id, _) => id,
            EntryStructCtor(id, _) => id,
            EntryLifetime(id, _) => id,
            EntryTyParam(id, _) => id,
            _ => return None
        })
    }

    fn to_node(self) -> Option<Node<'ast>> {
        Some(match self {
            EntryItem(_, n) => NodeItem(n),
            EntryForeignItem(_, n) => NodeForeignItem(n),
            EntryTraitItem(_, n) => NodeTraitItem(n),
            EntryImplItem(_, n) => NodeImplItem(n),
            EntryVariant(_, n) => NodeVariant(n),
            EntryExpr(_, n) => NodeExpr(n),
            EntryStmt(_, n) => NodeStmt(n),
            EntryLocal(_, n) => NodeLocal(n),
            EntryPat(_, n) => NodePat(n),
            EntryBlock(_, n) => NodeBlock(n),
            EntryStructCtor(_, n) => NodeStructCtor(n),
            EntryLifetime(_, n) => NodeLifetime(n),
            EntryTyParam(_, n) => NodeTyParam(n),
            _ => return None
        })
    }
}

/// Stores a crate and any number of inlined items from other crates.
pub struct Forest {
    krate: Crate,
    pub dep_graph: DepGraph,
    inlined_items: TypedArena<InlinedItem>
}

impl Forest {
    pub fn new(krate: Crate, dep_graph: &DepGraph) -> Forest {
        Forest {
            krate: krate,
            dep_graph: dep_graph.clone(),
            inlined_items: TypedArena::new()
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
    map: RefCell<Vec<MapEntry<'ast>>>,

    definitions: RefCell<Definitions>,
}

impl<'ast> Map<'ast> {
    /// Registers a read in the dependency graph of the AST node with
    /// the given `id`. This needs to be called each time a public
    /// function returns the HIR for a node -- in other words, when it
    /// "reveals" the content of a node to the caller (who might not
    /// otherwise have had access to those contents, and hence needs a
    /// read recorded). If the function just returns a DefId or
    /// NodeId, no actual content was returned, so no read is needed.
    fn read(&self, id: NodeId) {
        self.dep_graph.read(self.dep_node(id));
    }

    fn dep_node(&self, id0: NodeId) -> DepNode<DefId> {
        let map = self.map.borrow();
        let mut id = id0;
        loop {
            match map[id as usize] {
                EntryItem(_, item) => {
                    let def_id = self.local_def_id(item.id);
                    // NB                          ^~~~~~~
                    //
                    // You would expect that `item.id == id`, but this
                    // is not always the case. In particular, for a
                    // ViewPath item like `use self::{mem, foo}`, we
                    // map the ids for `mem` and `foo` to the
                    // enclosing view path item. This seems mega super
                    // ultra wrong, but then who am I to judge?
                    // -nmatsakis
                    return DepNode::Hir(def_id);
                }

                EntryForeignItem(p, _) |
                EntryTraitItem(p, _) |
                EntryImplItem(p, _) |
                EntryVariant(p, _) |
                EntryExpr(p, _) |
                EntryStmt(p, _) |
                EntryLocal(p, _) |
                EntryPat(p, _) |
                EntryBlock(p, _) |
                EntryStructCtor(p, _) |
                EntryLifetime(p, _) |
                EntryTyParam(p, _) =>
                    id = p,

                RootCrate |
                RootInlinedParent(_) =>
                    // FIXME(#32015) clarify story about cross-crate dep tracking
                    return DepNode::Krate,

                NotPresent =>
                    // Some nodes, notably struct fields, are not
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
        self.definitions.borrow().len()
    }

    pub fn def_key(&self, def_id: DefId) -> DefKey {
        assert!(def_id.is_local());
        self.definitions.borrow().def_key(def_id.index)
    }

    pub fn def_path_from_id(&self, id: NodeId) -> Option<DefPath> {
        self.opt_local_def_id(id).map(|def_id| {
            self.def_path(def_id)
        })
    }

    pub fn def_path(&self, def_id: DefId) -> DefPath {
        assert!(def_id.is_local());
        self.definitions.borrow().def_path(def_id.index)
    }

    pub fn def_index_for_def_key(&self, def_key: DefKey) -> Option<DefIndex> {
        self.definitions.borrow().def_index_for_def_key(def_key)
    }

    pub fn local_def_id(&self, node: NodeId) -> DefId {
        self.opt_local_def_id(node).unwrap_or_else(|| {
            bug!("local_def_id: no entry for `{}`, which has a map of `{:?}`",
                 node, self.find_entry(node))
        })
    }

    pub fn opt_local_def_id(&self, node: NodeId) -> Option<DefId> {
        self.definitions.borrow().opt_local_def_id(node)
    }

    pub fn as_local_node_id(&self, def_id: DefId) -> Option<NodeId> {
        self.definitions.borrow().as_local_node_id(def_id)
    }

    fn entry_count(&self) -> usize {
        self.map.borrow().len()
    }

    fn find_entry(&self, id: NodeId) -> Option<MapEntry<'ast>> {
        self.map.borrow().get(id as usize).cloned()
    }

    pub fn krate(&self) -> &'ast Crate {
        self.forest.krate()
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
            if parent_node == 0 {
                return Ok(0);
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
        let parent = self.get_parent(id);
        match self.find_entry(parent) {
            Some(RootInlinedParent(&II::TraitItem(did, _))) |
            Some(RootInlinedParent(&II::ImplItem(did, _))) => did,
            _ => self.local_def_id(parent)
        }
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
            /// Wrong but OK, because the only inlined foreign items are intrinsics.
            Some(RootInlinedParent(_)) => Some(Abi::RustIntrinsic),
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

    pub fn expect_trait_item(&self, id: NodeId) -> &'ast TraitItem {
        match self.find(id) {
            Some(NodeTraitItem(item)) => item,
            _ => bug!("expected trait item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_struct(&self, id: NodeId) -> &'ast VariantData {
        match self.find(id) {
            Some(NodeItem(i)) => {
                match i.node {
                    ItemStruct(ref struct_def, _) => struct_def,
                    _ => bug!("struct ID bound to non-struct")
                }
            }
            Some(NodeVariant(variant)) => {
                if variant.node.data.is_struct() {
                    &variant.node.data
                } else {
                    bug!("struct ID bound to enum variant that isn't struct-like")
                }
            }
            _ => bug!("expected struct, found {}", self.node_to_string(id)),
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

    /// Returns the name associated with the given NodeId's AST.
    pub fn name(&self, id: NodeId) -> Name {
        match self.get(id) {
            NodeItem(i) => i.name,
            NodeForeignItem(i) => i.name,
            NodeImplItem(ii) => ii.name,
            NodeTraitItem(ti) => ti.name,
            NodeVariant(v) => v.node.name,
            NodeLifetime(lt) => lt.name,
            NodeTyParam(tp) => tp.name,
            NodeLocal(&Pat { node: PatKind::Binding(_,l,_), .. }) => l.node,
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
            idx: 0,
        }
    }

    pub fn opt_span(&self, id: NodeId) -> Option<Span> {
        let sp = match self.find(id) {
            Some(NodeItem(item)) => item.span,
            Some(NodeForeignItem(foreign_item)) => foreign_item.span,
            Some(NodeTraitItem(trait_method)) => trait_method.span,
            Some(NodeImplItem(ref impl_item)) => impl_item.span,
            Some(NodeVariant(variant)) => variant.span,
            Some(NodeExpr(expr)) => expr.span,
            Some(NodeStmt(stmt)) => stmt.span,
            Some(NodeLocal(pat)) => pat.span,
            Some(NodePat(pat)) => pat.span,
            Some(NodeBlock(block)) => block.span,
            Some(NodeStructCtor(_)) => self.expect_item(self.get_parent(id)).span,
            Some(NodeTyParam(ty_param)) => ty_param.span,
            _ => return None,
        };
        Some(sp)
    }

    pub fn span(&self, id: NodeId) -> Span {
        self.read(id); // reveals span from node
        self.opt_span(id)
            .unwrap_or_else(|| bug!("AstMap.span: could not find span for id {:?}", id))
    }

    pub fn span_if_local(&self, id: DefId) -> Option<Span> {
        self.as_local_node_id(id).map(|id| self.span(id))
    }

    pub fn def_id_span(&self, def_id: DefId, fallback: Span) -> Span {
        if let Some(node_id) = self.as_local_node_id(def_id) {
            self.opt_span(node_id).unwrap_or(fallback)
        } else {
            fallback
        }
    }

    pub fn node_to_string(&self, id: NodeId) -> String {
        node_id_to_string(self, id, true)
    }

    pub fn node_to_user_string(&self, id: NodeId) -> String {
        node_id_to_string(self, id, false)
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
            if &part[..] != mod_name.as_str() {
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
        name.as_str() == &self.item_name[..] &&
            self.suffix_matches(parent_of_n)
    }
}

impl<'a, 'ast> Iterator for NodesMatchingSuffix<'a, 'ast> {
    type Item = NodeId;

    fn next(&mut self) -> Option<NodeId> {
        loop {
            let idx = self.idx;
            if idx as usize >= self.map.entry_count() {
                return None;
            }
            self.idx += 1;
            let name = match self.map.find_entry(idx) {
                Some(EntryItem(_, n))       => n.name(),
                Some(EntryForeignItem(_, n))=> n.name(),
                Some(EntryTraitItem(_, n))  => n.name(),
                Some(EntryImplItem(_, n))   => n.name(),
                Some(EntryVariant(_, n))    => n.name(),
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
impl Named for TraitItem { fn name(&self) -> Name { self.name } }
impl Named for ImplItem { fn name(&self) -> Name { self.name } }

pub trait FoldOps {
    fn new_id(&self, id: NodeId) -> NodeId {
        id
    }
    fn new_def_id(&self, def_id: DefId) -> DefId {
        def_id
    }
    fn new_span(&self, span: Span) -> Span {
        span
    }
}

/// A Folder that updates IDs and Span's according to fold_ops.
struct IdAndSpanUpdater<F> {
    fold_ops: F
}

impl<F: FoldOps> Folder for IdAndSpanUpdater<F> {
    fn new_id(&mut self, id: NodeId) -> NodeId {
        self.fold_ops.new_id(id)
    }

    fn new_span(&mut self, span: Span) -> Span {
        self.fold_ops.new_span(span)
    }
}

pub fn collect_definitions<'ast>(krate: &'ast ast::Crate) -> Definitions {
    let mut def_collector = DefCollector::root();
    visit::walk_crate(&mut def_collector, krate);
    def_collector.definitions
}

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
        map: RefCell::new(map),
        definitions: RefCell::new(definitions),
    }
}

/// Used for items loaded from external crate that are being inlined into this
/// crate.
pub fn map_decoded_item<'ast, F: FoldOps>(map: &Map<'ast>,
                                          parent_def_path: DefPath,
                                          parent_def_id: DefId,
                                          ii: InlinedItem,
                                          fold_ops: F)
                                          -> &'ast InlinedItem {
    let mut fld = IdAndSpanUpdater { fold_ops: fold_ops };
    let ii = match ii {
        II::Item(i) => II::Item(i.map(|i| fld.fold_item(i))),
        II::TraitItem(d, ti) => {
            II::TraitItem(fld.fold_ops.new_def_id(d),
                          ti.map(|ti| fld.fold_trait_item(ti)))
        }
        II::ImplItem(d, ii) => {
            II::ImplItem(fld.fold_ops.new_def_id(d),
                         ii.map(|ii| fld.fold_impl_item(ii)))
        }
        II::Foreign(i) => II::Foreign(i.map(|i| fld.fold_foreign_item(i)))
    };

    let ii = map.forest.inlined_items.alloc(ii);
    let ii_parent_id = fld.new_id(DUMMY_NODE_ID);

    let defs = mem::replace(&mut *map.definitions.borrow_mut(), Definitions::new());
    let mut def_collector = DefCollector::extend(ii_parent_id,
                                                 parent_def_path.clone(),
                                                 parent_def_id,
                                                 defs);
    def_collector.walk_item(ii, map.krate());
    *map.definitions.borrow_mut() = def_collector.definitions;

    let mut collector = NodeCollector::extend(map.krate(),
                                              ii,
                                              ii_parent_id,
                                              parent_def_path,
                                              parent_def_id,
                                              mem::replace(&mut *map.map.borrow_mut(), vec![]));
    ii.visit(&mut collector);
    *map.map.borrow_mut() = collector.map;

    ii
}

pub trait NodePrinter {
    fn print_node(&mut self, node: &Node) -> io::Result<()>;
}

impl<'a> NodePrinter for pprust::State<'a> {
    fn print_node(&mut self, node: &Node) -> io::Result<()> {
        match *node {
            NodeItem(a)        => self.print_item(&a),
            NodeForeignItem(a) => self.print_foreign_item(&a),
            NodeTraitItem(a)   => self.print_trait_item(a),
            NodeImplItem(a)    => self.print_impl_item(a),
            NodeVariant(a)     => self.print_variant(&a),
            NodeExpr(a)        => self.print_expr(&a),
            NodeStmt(a)        => self.print_stmt(&a),
            NodePat(a)         => self.print_pat(&a),
            NodeBlock(a)       => self.print_block(&a),
            NodeLifetime(a)    => self.print_lifetime(&a),
            NodeTyParam(_)     => bug!("cannot print TyParam"),
            // these cases do not carry enough information in the
            // ast_map to reconstruct their full structure for pretty
            // printing.
            NodeLocal(_)       => bug!("cannot print isolated Local"),
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
                ConstTraitItem(..) => "assoc constant",
                MethodTraitItem(..) => "trait method",
                TypeTraitItem(..) => "assoc type",
            };

            format!("{} {} in {}{}", kind, ti.name, path_str(), id_str)
        }
        Some(NodeVariant(ref variant)) => {
            format!("variant {} in {}{}",
                    variant.node.name,
                    path_str(), id_str)
        }
        Some(NodeExpr(ref expr)) => {
            format!("expr {}{}", pprust::expr_to_string(&expr), id_str)
        }
        Some(NodeStmt(ref stmt)) => {
            format!("stmt {}{}", pprust::stmt_to_string(&stmt), id_str)
        }
        Some(NodeLocal(ref pat)) => {
            format!("local {}{}", pprust::pat_to_string(&pat), id_str)
        }
        Some(NodePat(ref pat)) => {
            format!("pat {}{}", pprust::pat_to_string(&pat), id_str)
        }
        Some(NodeBlock(ref block)) => {
            format!("block {}{}", pprust::block_to_string(&block), id_str)
        }
        Some(NodeStructCtor(_)) => {
            format!("struct_ctor {}{}", path_str(), id_str)
        }
        Some(NodeLifetime(ref l)) => {
            format!("lifetime {}{}",
                    pprust::lifetime_to_string(&l), id_str)
        }
        Some(NodeTyParam(ref ty_param)) => {
            format!("typaram {:?}{}", ty_param, id_str)
        }
        None => {
            format!("unknown node{}", id_str)
        }
    }
}
