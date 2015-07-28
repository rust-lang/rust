// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::Node::*;
pub use self::PathElem::*;
use self::MapEntry::*;

use syntax::abi;
use syntax::ast::*;
use syntax::ast_util;
use syntax::codemap::{DUMMY_SP, Span, Spanned};
use syntax::fold::Folder;
use syntax::parse::token;
use syntax::print::pprust;
use syntax::visit::{self, Visitor};

use arena::TypedArena;
use std::cell::RefCell;
use std::fmt;
use std::io;
use std::iter::{self, repeat};
use std::mem;
use std::slice;

pub mod blocks;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PathElem {
    PathMod(Name),
    PathName(Name)
}

impl PathElem {
    pub fn name(&self) -> Name {
        match *self {
            PathMod(name) | PathName(name) => name
        }
    }
}

impl fmt::Display for PathElem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Clone)]
pub struct LinkedPathNode<'a> {
    node: PathElem,
    next: LinkedPath<'a>,
}

#[derive(Copy, Clone)]
pub struct LinkedPath<'a>(Option<&'a LinkedPathNode<'a>>);

impl<'a> LinkedPath<'a> {
    pub fn empty() -> LinkedPath<'a> {
        LinkedPath(None)
    }

    pub fn from(node: &'a LinkedPathNode) -> LinkedPath<'a> {
        LinkedPath(Some(node))
    }
}

impl<'a> Iterator for LinkedPath<'a> {
    type Item = PathElem;

    fn next(&mut self) -> Option<PathElem> {
        match self.0 {
            Some(node) => {
                *self = node.next;
                Some(node.node)
            }
            None => None
        }
    }
}

/// The type of the iterator used by with_path.
pub type PathElems<'a, 'b> = iter::Chain<iter::Cloned<slice::Iter<'a, PathElem>>, LinkedPath<'b>>;

pub fn path_to_string<PI: Iterator<Item=PathElem>>(path: PI) -> String {
    let itr = token::get_ident_interner();

    path.fold(String::new(), |mut s, e| {
        let e = itr.get(e.name());
        if !s.is_empty() {
            s.push_str("::");
        }
        s.push_str(&e[..]);
        s
    })
}

#[derive(Copy, Clone, Debug)]
pub enum Node<'ast> {
    NodeItem(&'ast Item),
    NodeForeignItem(&'ast ForeignItem),
    NodeTraitItem(&'ast TraitItem),
    NodeImplItem(&'ast ImplItem),
    NodeVariant(&'ast Variant),
    NodeExpr(&'ast Expr),
    NodeStmt(&'ast Stmt),
    NodeArg(&'ast Pat),
    NodeLocal(&'ast Pat),
    NodePat(&'ast Pat),
    NodeBlock(&'ast Block),

    /// NodeStructCtor represents a tuple struct.
    NodeStructCtor(&'ast StructDef),

    NodeLifetime(&'ast Lifetime),
    NodeTyParam(&'ast TyParam)
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
    EntryExpr(NodeId, &'ast Expr),
    EntryStmt(NodeId, &'ast Stmt),
    EntryArg(NodeId, &'ast Pat),
    EntryLocal(NodeId, &'ast Pat),
    EntryPat(NodeId, &'ast Pat),
    EntryBlock(NodeId, &'ast Block),
    EntryStructCtor(NodeId, &'ast StructDef),
    EntryLifetime(NodeId, &'ast Lifetime),
    EntryTyParam(NodeId, &'ast TyParam),

    /// Roots for node trees.
    RootCrate,
    RootInlinedParent(&'ast InlinedParent)
}

impl<'ast> Clone for MapEntry<'ast> {
    fn clone(&self) -> MapEntry<'ast> {
        *self
    }
}

#[derive(Debug)]
struct InlinedParent {
    path: Vec<PathElem>,
    ii: InlinedItem
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
            NodeArg(n) => EntryArg(p, n),
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
            EntryArg(id, _) => id,
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
            EntryArg(_, n) => NodeArg(n),
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
    inlined_items: TypedArena<InlinedParent>
}

impl Forest {
    pub fn new(krate: Crate) -> Forest {
        Forest {
            krate: krate,
            inlined_items: TypedArena::new()
        }
    }

    pub fn krate<'ast>(&'ast self) -> &'ast Crate {
        &self.krate
    }
}

/// Represents a mapping from Node IDs to AST elements and their parent
/// Node IDs
pub struct Map<'ast> {
    /// The backing storage for all the AST nodes.
    forest: &'ast Forest,

    /// NodeIds are sequential integers from 0, so we can be
    /// super-compact by storing them in a vector. Not everything with
    /// a NodeId is in the map, but empirically the occupancy is about
    /// 75-80%, so there's not too much overhead (certainly less than
    /// a hashmap, since they (at the time of writing) have a maximum
    /// of 75% occupancy).
    ///
    /// Also, indexing is pretty quick when you've got a vector and
    /// plain old integers.
    map: RefCell<Vec<MapEntry<'ast>>>
}

impl<'ast> Map<'ast> {
    fn entry_count(&self) -> usize {
        self.map.borrow().len()
    }

    fn find_entry(&self, id: NodeId) -> Option<MapEntry<'ast>> {
        self.map.borrow().get(id as usize).cloned()
    }

    pub fn krate(&self) -> &'ast Crate {
        &self.forest.krate
    }

    /// Retrieve the Node corresponding to `id`, panicking if it cannot
    /// be found.
    pub fn get(&self, id: NodeId) -> Node<'ast> {
        match self.find(id) {
            Some(node) => node,
            None => panic!("couldn't find node id {} in the AST map", id)
        }
    }

    /// Retrieve the Node corresponding to `id`, returning None if
    /// cannot be found.
    pub fn find(&self, id: NodeId) -> Option<Node<'ast>> {
        self.find_entry(id).and_then(|x| x.to_node())
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
            Some(RootInlinedParent(&InlinedParent {ii: IITraitItem(did, _), ..})) => did,
            Some(RootInlinedParent(&InlinedParent {ii: IIImplItem(did, _), ..})) => did,
            _ => ast_util::local_def(parent)
        }
    }

    pub fn get_foreign_abi(&self, id: NodeId) -> abi::Abi {
        let parent = self.get_parent(id);
        let abi = match self.find_entry(parent) {
            Some(EntryItem(_, i)) => {
                match i.node {
                    ItemForeignMod(ref nm) => Some(nm.abi),
                    _ => None
                }
            }
            /// Wrong but OK, because the only inlined foreign items are intrinsics.
            Some(RootInlinedParent(_)) => Some(abi::RustIntrinsic),
            _ => None
        };
        match abi {
            Some(abi) => abi,
            None => panic!("expected foreign mod or inlined parent, found {}",
                          self.node_to_string(parent))
        }
    }

    pub fn get_foreign_vis(&self, id: NodeId) -> Visibility {
        let vis = self.expect_foreign_item(id).vis;
        match self.find(self.get_parent(id)) {
            Some(NodeItem(i)) => vis.inherit_from(i.vis),
            _ => vis
        }
    }

    pub fn expect_item(&self, id: NodeId) -> &'ast Item {
        match self.find(id) {
            Some(NodeItem(item)) => item,
            _ => panic!("expected item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_trait_item(&self, id: NodeId) -> &'ast TraitItem {
        match self.find(id) {
            Some(NodeTraitItem(item)) => item,
            _ => panic!("expected trait item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_struct(&self, id: NodeId) -> &'ast StructDef {
        match self.find(id) {
            Some(NodeItem(i)) => {
                match i.node {
                    ItemStruct(ref struct_def, _) => &**struct_def,
                    _ => panic!("struct ID bound to non-struct")
                }
            }
            Some(NodeVariant(variant)) => {
                match variant.node.kind {
                    StructVariantKind(ref struct_def) => &**struct_def,
                    _ => panic!("struct ID bound to enum variant that isn't struct-like"),
                }
            }
            _ => panic!(format!("expected struct, found {}", self.node_to_string(id))),
        }
    }

    pub fn expect_variant(&self, id: NodeId) -> &'ast Variant {
        match self.find(id) {
            Some(NodeVariant(variant)) => variant,
            _ => panic!(format!("expected variant, found {}", self.node_to_string(id))),
        }
    }

    pub fn expect_foreign_item(&self, id: NodeId) -> &'ast ForeignItem {
        match self.find(id) {
            Some(NodeForeignItem(item)) => item,
            _ => panic!("expected foreign item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_expr(&self, id: NodeId) -> &'ast Expr {
        match self.find(id) {
            Some(NodeExpr(expr)) => expr,
            _ => panic!("expected expr, found {}", self.node_to_string(id))
        }
    }

    /// returns the name associated with the given NodeId's AST
    pub fn get_path_elem(&self, id: NodeId) -> PathElem {
        let node = self.get(id);
        match node {
            NodeItem(item) => {
                match item.node {
                    ItemMod(_) | ItemForeignMod(_) => {
                        PathMod(item.ident.name)
                    }
                    _ => PathName(item.ident.name)
                }
            }
            NodeForeignItem(i) => PathName(i.ident.name),
            NodeImplItem(ii) => PathName(ii.ident.name),
            NodeTraitItem(ti) => PathName(ti.ident.name),
            NodeVariant(v) => PathName(v.node.name.name),
            _ => panic!("no path elem for {:?}", node)
        }
    }

    pub fn with_path<T, F>(&self, id: NodeId, f: F) -> T where
        F: FnOnce(PathElems) -> T,
    {
        self.with_path_next(id, LinkedPath::empty(), f)
    }

    pub fn path_to_string(&self, id: NodeId) -> String {
        self.with_path(id, |path| path_to_string(path))
    }

    fn path_to_str_with_ident(&self, id: NodeId, i: Ident) -> String {
        self.with_path(id, |path| {
            path_to_string(path.chain(Some(PathName(i.name))))
        })
    }

    fn with_path_next<T, F>(&self, id: NodeId, next: LinkedPath, f: F) -> T where
        F: FnOnce(PathElems) -> T,
    {
        let parent = self.get_parent(id);
        let parent = match self.find_entry(id) {
            Some(EntryForeignItem(..)) => {
                // Anonymous extern items go in the parent scope.
                self.get_parent(parent)
            }
            // But tuple struct ctors don't have names, so use the path of its
            // parent, the struct item. Similarly with closure expressions.
            Some(EntryStructCtor(..)) | Some(EntryExpr(..)) => {
                return self.with_path_next(parent, next, f);
            }
            _ => parent
        };
        if parent == id {
            match self.find_entry(id) {
                Some(RootInlinedParent(data)) => {
                    f(data.path.iter().cloned().chain(next))
                }
                _ => f([].iter().cloned().chain(next))
            }
        } else {
            self.with_path_next(parent, LinkedPath::from(&LinkedPathNode {
                node: self.get_path_elem(id),
                next: next
            }), f)
        }
    }

    /// Given a node ID, get a list of of attributes associated with the AST
    /// corresponding to the Node ID
    pub fn attrs(&self, id: NodeId) -> &'ast [Attribute] {
        let attrs = match self.find(id) {
            Some(NodeItem(i)) => Some(&i.attrs[..]),
            Some(NodeForeignItem(fi)) => Some(&fi.attrs[..]),
            Some(NodeTraitItem(ref ti)) => Some(&ti.attrs[..]),
            Some(NodeImplItem(ref ii)) => Some(&ii.attrs[..]),
            Some(NodeVariant(ref v)) => Some(&v.node.attrs[..]),
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
            Some(NodeArg(pat)) | Some(NodeLocal(pat)) => pat.span,
            Some(NodePat(pat)) => pat.span,
            Some(NodeBlock(block)) => block.span,
            Some(NodeStructCtor(_)) => self.expect_item(self.get_parent(id)).span,
            Some(NodeTyParam(ty_param)) => ty_param.span,
            _ => return None,
        };
        Some(sp)
    }

    pub fn span(&self, id: NodeId) -> Span {
        self.opt_span(id)
            .unwrap_or_else(|| panic!("AstMap.span: could not find span for id {:?}", id))
    }

    pub fn def_id_span(&self, def_id: DefId, fallback: Span) -> Span {
        if def_id.krate == LOCAL_CRATE {
            self.opt_span(def_id.node).unwrap_or(fallback)
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
                    Some(NodeItem(item)) if item_is_mod(&*item) =>
                        return Some((id, item.ident.name)),
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

impl Named for Item { fn name(&self) -> Name { self.ident.name } }
impl Named for ForeignItem { fn name(&self) -> Name { self.ident.name } }
impl Named for Variant_ { fn name(&self) -> Name { self.name.name } }
impl Named for TraitItem { fn name(&self) -> Name { self.ident.name } }
impl Named for ImplItem { fn name(&self) -> Name { self.ident.name } }

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

/// A Visitor that walks over an AST and collects Node's into an AST Map.
struct NodeCollector<'ast> {
    map: Vec<MapEntry<'ast>>,
    parent_node: NodeId,
}

impl<'ast> NodeCollector<'ast> {
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

        match i.node {
            ItemImpl(_, _, _, _, _, ref impl_items) => {
                for ii in impl_items {
                    self.insert(ii.id, NodeImplItem(ii));
                }
            }
            ItemEnum(ref enum_definition, _) => {
                for v in &enum_definition.variants {
                    self.insert(v.node.id, NodeVariant(&**v));
                }
            }
            ItemForeignMod(ref nm) => {
                for nitem in &nm.items {
                    self.insert(nitem.id, NodeForeignItem(&**nitem));
                }
            }
            ItemStruct(ref struct_def, _) => {
                // If this is a tuple-like struct, register the constructor.
                match struct_def.ctor_id {
                    Some(ctor_id) => {
                        self.insert(ctor_id, NodeStructCtor(&**struct_def));
                    }
                    None => {}
                }
            }
            ItemTrait(_, _, ref bounds, ref trait_items) => {
                for b in bounds.iter() {
                    if let TraitTyParamBound(ref t, TraitBoundModifier::None) = *b {
                        self.insert(t.trait_ref.ref_id, NodeItem(i));
                    }
                }

                for ti in trait_items {
                    self.insert(ti.id, NodeTraitItem(ti));
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

    fn visit_generics(&mut self, generics: &'ast Generics) {
        for ty_param in generics.ty_params.iter() {
            self.insert(ty_param.id, NodeTyParam(ty_param));
        }

        visit::walk_generics(self, generics);
    }

    fn visit_trait_item(&mut self, ti: &'ast TraitItem) {
        let parent_node = self.parent_node;
        self.parent_node = ti.id;
        visit::walk_trait_item(self, ti);
        self.parent_node = parent_node;
    }

    fn visit_impl_item(&mut self, ii: &'ast ImplItem) {
        let parent_node = self.parent_node;
        self.parent_node = ii.id;

        visit::walk_impl_item(self, ii);

        self.parent_node = parent_node;
    }

    fn visit_pat(&mut self, pat: &'ast Pat) {
        self.insert(pat.id, match pat.node {
            // Note: this is at least *potentially* a pattern...
            PatIdent(..) => NodeLocal(pat),
            _ => NodePat(pat)
        });

        let parent_node = self.parent_node;
        self.parent_node = pat.id;
        visit::walk_pat(self, pat);
        self.parent_node = parent_node;
    }

    fn visit_expr(&mut self, expr: &'ast Expr) {
        self.insert(expr.id, NodeExpr(expr));
        let parent_node = self.parent_node;
        self.parent_node = expr.id;
        visit::walk_expr(self, expr);
        self.parent_node = parent_node;
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt) {
        let id = ast_util::stmt_id(stmt);
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

    fn visit_lifetime_ref(&mut self, lifetime: &'ast Lifetime) {
        self.insert(lifetime.id, NodeLifetime(lifetime));
    }

    fn visit_lifetime_def(&mut self, def: &'ast LifetimeDef) {
        self.visit_lifetime_ref(&def.lifetime);
    }
}

pub fn map_crate<'ast, F: FoldOps>(forest: &'ast mut Forest, fold_ops: F) -> Map<'ast> {
    // Replace the crate with an empty one to take it out.
    let krate = mem::replace(&mut forest.krate, Crate {
        module: Mod {
            inner: DUMMY_SP,
            items: vec![],
        },
        attrs: vec![],
        config: vec![],
        exported_macros: vec![],
        span: DUMMY_SP
    });
    forest.krate = IdAndSpanUpdater { fold_ops: fold_ops }.fold_crate(krate);

    let mut collector = NodeCollector {
        map: vec![],
        parent_node: CRATE_NODE_ID,
    };
    collector.insert_entry(CRATE_NODE_ID, RootCrate);
    visit::walk_crate(&mut collector, &forest.krate);
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
        map: RefCell::new(map)
    }
}

/// Used for items loaded from external crate that are being inlined into this
/// crate.  The `path` should be the path to the item but should not include
/// the item itself.
pub fn map_decoded_item<'ast, F: FoldOps>(map: &Map<'ast>,
                                          path: Vec<PathElem>,
                                          ii: InlinedItem,
                                          fold_ops: F)
                                          -> &'ast InlinedItem {
    let mut fld = IdAndSpanUpdater { fold_ops: fold_ops };
    let ii = match ii {
        IIItem(i) => IIItem(fld.fold_item(i).expect_one("expected one item")),
        IITraitItem(d, ti) => {
            IITraitItem(fld.fold_ops.new_def_id(d),
                        fld.fold_trait_item(ti).expect_one("expected one trait item"))
        }
        IIImplItem(d, ii) => {
            IIImplItem(fld.fold_ops.new_def_id(d),
                       fld.fold_impl_item(ii).expect_one("expected one impl item"))
        }
        IIForeign(i) => IIForeign(fld.fold_foreign_item(i))
    };

    let ii_parent = map.forest.inlined_items.alloc(InlinedParent {
        path: path,
        ii: ii
    });

    let ii_parent_id = fld.new_id(DUMMY_NODE_ID);
    let mut collector = NodeCollector {
        map: mem::replace(&mut *map.map.borrow_mut(), vec![]),
        parent_node: ii_parent_id,
    };
    collector.insert_entry(ii_parent_id, RootInlinedParent(ii_parent));
    visit::walk_inlined_item(&mut collector, &ii_parent.ii);

    // Methods get added to the AST map when their impl is visited.  Since we
    // don't decode and instantiate the impl, but just the method, we have to
    // add it to the table now. Likewise with foreign items.
    match ii_parent.ii {
        IIItem(_) => {}
        IITraitItem(_, ref ti) => {
            collector.insert(ti.id, NodeTraitItem(ti));
        }
        IIImplItem(_, ref ii) => {
            collector.insert(ii.id, NodeImplItem(ii));
        }
        IIForeign(ref i) => {
            collector.insert(i.id, NodeForeignItem(i));
        }
    }
    *map.map.borrow_mut() = collector.map;
    &ii_parent.ii
}

pub trait NodePrinter {
    fn print_node(&mut self, node: &Node) -> io::Result<()>;
}

impl<'a> NodePrinter for pprust::State<'a> {
    fn print_node(&mut self, node: &Node) -> io::Result<()> {
        match *node {
            NodeItem(a)        => self.print_item(&*a),
            NodeForeignItem(a) => self.print_foreign_item(&*a),
            NodeTraitItem(a)   => self.print_trait_item(a),
            NodeImplItem(a)    => self.print_impl_item(a),
            NodeVariant(a)     => self.print_variant(&*a),
            NodeExpr(a)        => self.print_expr(&*a),
            NodeStmt(a)        => self.print_stmt(&*a),
            NodePat(a)         => self.print_pat(&*a),
            NodeBlock(a)       => self.print_block(&*a),
            NodeLifetime(a)    => self.print_lifetime(&*a),
            NodeTyParam(_)     => panic!("cannot print TyParam"),
            // these cases do not carry enough information in the
            // ast_map to reconstruct their full structure for pretty
            // printing.
            NodeLocal(_)       => panic!("cannot print isolated Local"),
            NodeArg(_)         => panic!("cannot print isolated Arg"),
            NodeStructCtor(_)  => panic!("cannot print isolated StructCtor"),
        }
    }
}

fn node_id_to_string(map: &Map, id: NodeId, include_id: bool) -> String {
    let id_str = format!(" (id={})", id);
    let id_str = if include_id { &id_str[..] } else { "" };

    match map.find(id) {
        Some(NodeItem(item)) => {
            let path_str = map.path_to_str_with_ident(id, item.ident);
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
                ItemMac(..) => "macro"
            };
            format!("{} {}{}", item_str, path_str, id_str)
        }
        Some(NodeForeignItem(item)) => {
            let path_str = map.path_to_str_with_ident(id, item.ident);
            format!("foreign item {}{}", path_str, id_str)
        }
        Some(NodeImplItem(ii)) => {
            match ii.node {
                ConstImplItem(..) => {
                    format!("assoc const {} in {}{}",
                            ii.ident,
                            map.path_to_string(id),
                            id_str)
                }
                MethodImplItem(..) => {
                    format!("method {} in {}{}",
                            ii.ident,
                            map.path_to_string(id), id_str)
                }
                TypeImplItem(_) => {
                    format!("assoc type {} in {}{}",
                            ii.ident,
                            map.path_to_string(id),
                            id_str)
                }
                MacImplItem(ref mac) => {
                    format!("method macro {}{}",
                            pprust::mac_to_string(mac), id_str)
                }
            }
        }
        Some(NodeTraitItem(ti)) => {
            let kind = match ti.node {
                ConstTraitItem(..) => "assoc constant",
                MethodTraitItem(..) => "trait method",
                TypeTraitItem(..) => "assoc type",
            };

            format!("{} {} in {}{}",
                    kind,
                    ti.ident,
                    map.path_to_string(id),
                    id_str)
        }
        Some(NodeVariant(ref variant)) => {
            format!("variant {} in {}{}",
                    variant.node.name,
                    map.path_to_string(id), id_str)
        }
        Some(NodeExpr(ref expr)) => {
            format!("expr {}{}", pprust::expr_to_string(&**expr), id_str)
        }
        Some(NodeStmt(ref stmt)) => {
            format!("stmt {}{}", pprust::stmt_to_string(&**stmt), id_str)
        }
        Some(NodeArg(ref pat)) => {
            format!("arg {}{}", pprust::pat_to_string(&**pat), id_str)
        }
        Some(NodeLocal(ref pat)) => {
            format!("local {}{}", pprust::pat_to_string(&**pat), id_str)
        }
        Some(NodePat(ref pat)) => {
            format!("pat {}{}", pprust::pat_to_string(&**pat), id_str)
        }
        Some(NodeBlock(ref block)) => {
            format!("block {}{}", pprust::block_to_string(&**block), id_str)
        }
        Some(NodeStructCtor(_)) => {
            format!("struct_ctor {}{}", map.path_to_string(id), id_str)
        }
        Some(NodeLifetime(ref l)) => {
            format!("lifetime {}{}",
                    pprust::lifetime_to_string(&**l), id_str)
        }
        Some(NodeTyParam(ref ty_param)) => {
            format!("typaram {:?}{}", ty_param, id_str)
        }
        None => {
            format!("unknown node{}", id_str)
        }
    }
}
