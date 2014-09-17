// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi;
use ast::*;
use ast_util;
use codemap::{DUMMY_SP, Span, Spanned};
use fold::Folder;
use parse::token;
use print::pprust;
use ptr::P;
use visit::{mod, Visitor};

use arena::TypedArena;
use std::cell::RefCell;
use std::fmt;
use std::io::IoResult;
use std::iter;
use std::mem;
use std::slice;

pub mod blocks;

#[deriving(Clone, PartialEq)]
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

impl fmt::Show for PathElem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let slot = token::get_name(self.name());
        write!(f, "{}", slot)
    }
}

#[deriving(Clone)]
struct LinkedPathNode<'a> {
    node: PathElem,
    next: LinkedPath<'a>,
}

type LinkedPath<'a> = Option<&'a LinkedPathNode<'a>>;

impl<'a> Iterator<PathElem> for LinkedPath<'a> {
    fn next(&mut self) -> Option<PathElem> {
        match *self {
            Some(node) => {
                *self = node.next;
                Some(node.node)
            }
            None => None
        }
    }
}

// HACK(eddyb) move this into libstd (value wrapper for slice::Items).
#[deriving(Clone)]
pub struct Values<'a, T:'a>(pub slice::Items<'a, T>);

impl<'a, T: Copy> Iterator<T> for Values<'a, T> {
    fn next(&mut self) -> Option<T> {
        let &Values(ref mut items) = self;
        items.next().map(|&x| x)
    }
}

/// The type of the iterator used by with_path.
pub type PathElems<'a, 'b> = iter::Chain<Values<'a, PathElem>, LinkedPath<'b>>;

pub fn path_to_string<PI: Iterator<PathElem>>(mut path: PI) -> String {
    let itr = token::get_ident_interner();

    path.fold(String::new(), |mut s, e| {
        let e = itr.get(e.name());
        if !s.is_empty() {
            s.push_str("::");
        }
        s.push_str(e.as_slice());
        s
    }).to_string()
}

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
}

/// Represents an entry and its parent Node ID
/// The odd layout is to bring down the total size.
#[deriving(Show)]
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

    /// Roots for node trees.
    RootCrate,
    RootInlinedParent(&'ast InlinedParent)
}

impl<'ast> Clone for MapEntry<'ast> {
    fn clone(&self) -> MapEntry<'ast> {
        *self
    }
}

#[deriving(Show)]
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
            NodeLifetime(n) => EntryLifetime(p, n)
        }
    }

    fn parent(self) -> Option<NodeId> {
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
    fn entry_count(&self) -> uint {
        self.map.borrow().len()
    }

    fn find_entry(&self, id: NodeId) -> Option<MapEntry<'ast>> {
        self.map.borrow().as_slice().get(id as uint).map(|e| *e)
    }

    pub fn krate(&self) -> &'ast Crate {
        &self.forest.krate
    }

    /// Retrieve the Node corresponding to `id`, failing if it cannot
    /// be found.
    pub fn get(&self, id: NodeId) -> Node<'ast> {
        match self.find(id) {
            Some(node) => node,
            None => fail!("couldn't find node id {} in the AST map", id)
        }
    }

    /// Retrieve the Node corresponding to `id`, returning None if
    /// cannot be found.
    pub fn find(&self, id: NodeId) -> Option<Node<'ast>> {
        self.find_entry(id).and_then(|x| x.to_node())
    }

    /// Retrieve the parent NodeId for `id`, or `id` itself if no
    /// parent is registered in this map.
    pub fn get_parent(&self, id: NodeId) -> NodeId {
        self.find_entry(id).and_then(|x| x.parent()).unwrap_or(id)
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
            None => fail!("expected foreign mod or inlined parent, found {}",
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
            _ => fail!("expected item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_struct(&self, id: NodeId) -> &'ast StructDef {
        match self.find(id) {
            Some(NodeItem(i)) => {
                match i.node {
                    ItemStruct(ref struct_def, _) => &**struct_def,
                    _ => fail!("struct ID bound to non-struct")
                }
            }
            Some(NodeVariant(variant)) => {
                match variant.node.kind {
                    StructVariantKind(ref struct_def) => &**struct_def,
                    _ => fail!("struct ID bound to enum variant that isn't struct-like"),
                }
            }
            _ => fail!(format!("expected struct, found {}", self.node_to_string(id))),
        }
    }

    pub fn expect_variant(&self, id: NodeId) -> &'ast Variant {
        match self.find(id) {
            Some(NodeVariant(variant)) => variant,
            _ => fail!(format!("expected variant, found {}", self.node_to_string(id))),
        }
    }

    pub fn expect_foreign_item(&self, id: NodeId) -> &'ast ForeignItem {
        match self.find(id) {
            Some(NodeForeignItem(item)) => item,
            _ => fail!("expected foreign item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_expr(&self, id: NodeId) -> &'ast Expr {
        match self.find(id) {
            Some(NodeExpr(expr)) => expr,
            _ => fail!("expected expr, found {}", self.node_to_string(id))
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
            NodeImplItem(ii) => {
                match *ii {
                    MethodImplItem(ref m) => {
                        match m.node {
                            MethDecl(ident, _, _, _, _, _, _, _) => {
                                PathName(ident.name)
                            }
                            MethMac(_) => {
                                fail!("no path elem for {:?}", node)
                            }
                        }
                    }
                    TypeImplItem(ref t) => PathName(t.ident.name),
                }
            },
            NodeTraitItem(tm) => match *tm {
                RequiredMethod(ref m) => PathName(m.ident.name),
                ProvidedMethod(ref m) => {
                    match m.node {
                        MethDecl(ident, _, _, _, _, _, _, _) => {
                            PathName(ident.name)
                        }
                        MethMac(_) => fail!("no path elem for {:?}", node),
                    }
                }
                TypeTraitItem(ref m) => PathName(m.ident.name),
            },
            NodeVariant(v) => PathName(v.node.name.name),
            _ => fail!("no path elem for {:?}", node)
        }
    }

    pub fn with_path<T>(&self, id: NodeId, f: |PathElems| -> T) -> T {
        self.with_path_next(id, None, f)
    }

    pub fn path_to_string(&self, id: NodeId) -> String {
        self.with_path(id, |path| path_to_string(path))
    }

    fn path_to_str_with_ident(&self, id: NodeId, i: Ident) -> String {
        self.with_path(id, |path| {
            path_to_string(path.chain(Some(PathName(i.name)).into_iter()))
        })
    }

    fn with_path_next<T>(&self, id: NodeId, next: LinkedPath, f: |PathElems| -> T) -> T {
        let parent = self.get_parent(id);
        let parent = match self.find_entry(id) {
            Some(EntryForeignItem(..)) | Some(EntryVariant(..)) => {
                // Anonymous extern items, enum variants and struct ctors
                // go in the parent scope.
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
                    f(Values(data.path.iter()).chain(next))
                }
                _ => f(Values([].iter()).chain(next))
            }
        } else {
            self.with_path_next(parent, Some(&LinkedPathNode {
                node: self.get_path_elem(id),
                next: next
            }), f)
        }
    }

    /// Given a node ID and a closure, apply the closure to the array
    /// of attributes associated with the AST corresponding to the Node ID
    pub fn with_attrs<T>(&self, id: NodeId, f: |Option<&[Attribute]>| -> T) -> T {
        let attrs = match self.get(id) {
            NodeItem(i) => Some(i.attrs.as_slice()),
            NodeForeignItem(fi) => Some(fi.attrs.as_slice()),
            NodeTraitItem(ref tm) => match **tm {
                RequiredMethod(ref type_m) => Some(type_m.attrs.as_slice()),
                ProvidedMethod(ref m) => Some(m.attrs.as_slice()),
                TypeTraitItem(ref typ) => Some(typ.attrs.as_slice()),
            },
            NodeImplItem(ref ii) => {
                match **ii {
                    MethodImplItem(ref m) => Some(m.attrs.as_slice()),
                    TypeImplItem(ref t) => Some(t.attrs.as_slice()),
                }
            }
            NodeVariant(ref v) => Some(v.node.attrs.as_slice()),
            // unit/tuple structs take the attributes straight from
            // the struct definition.
            // FIXME(eddyb) make this work again (requires access to the map).
            NodeStructCtor(_) => {
                return self.with_attrs(self.get_parent(id), f);
            }
            _ => None
        };
        f(attrs)
    }

    /// Returns an iterator that yields the node id's with paths that
    /// match `parts`.  (Requires `parts` is non-empty.)
    ///
    /// For example, if given `parts` equal to `["bar", "quux"]`, then
    /// the iterator will produce node id's for items with paths
    /// such as `foo::bar::quux`, `bar::quux`, `other::bar::quux`, and
    /// any other such items it can find in the map.
    pub fn nodes_matching_suffix<'a, S:Str>(&'a self, parts: &'a [S])
                                 -> NodesMatchingSuffix<'a, 'ast, S> {
        NodesMatchingSuffix {
            map: self,
            item_name: parts.last().unwrap(),
            in_which: parts.slice_to(parts.len() - 1),
            idx: 0,
        }
    }

    pub fn opt_span(&self, id: NodeId) -> Option<Span> {
        let sp = match self.find(id) {
            Some(NodeItem(item)) => item.span,
            Some(NodeForeignItem(foreign_item)) => foreign_item.span,
            Some(NodeTraitItem(trait_method)) => {
                match *trait_method {
                    RequiredMethod(ref type_method) => type_method.span,
                    ProvidedMethod(ref method) => method.span,
                    TypeTraitItem(ref typedef) => typedef.span,
                }
            }
            Some(NodeImplItem(ref impl_item)) => {
                match **impl_item {
                    MethodImplItem(ref method) => method.span,
                    TypeImplItem(ref typedef) => typedef.span,
                }
            }
            Some(NodeVariant(variant)) => variant.span,
            Some(NodeExpr(expr)) => expr.span,
            Some(NodeStmt(stmt)) => stmt.span,
            Some(NodeArg(pat)) | Some(NodeLocal(pat)) => pat.span,
            Some(NodePat(pat)) => pat.span,
            Some(NodeBlock(block)) => block.span,
            Some(NodeStructCtor(_)) => self.expect_item(self.get_parent(id)).span,
            _ => return None,
        };
        Some(sp)
    }

    pub fn span(&self, id: NodeId) -> Span {
        self.opt_span(id)
            .unwrap_or_else(|| fail!("AstMap.span: could not find span for id {}", id))
    }

    pub fn node_to_string(&self, id: NodeId) -> String {
        node_id_to_string(self, id)
    }
}

pub struct NodesMatchingSuffix<'a, 'ast:'a, S:'a> {
    map: &'a Map<'ast>,
    item_name: &'a S,
    in_which: &'a [S],
    idx: NodeId,
}

impl<'a, 'ast, S:Str> NodesMatchingSuffix<'a, 'ast, S> {
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
            if part.as_slice() != mod_name.as_str() {
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
        name.as_str() == self.item_name.as_slice() &&
            self.suffix_matches(parent_of_n)
    }
}

impl<'a, 'ast, S:Str> Iterator<NodeId> for NodesMatchingSuffix<'a, 'ast, S> {
    fn next(&mut self) -> Option<NodeId> {
        loop {
            let idx = self.idx;
            if idx as uint >= self.map.entry_count() {
                return None;
            }
            self.idx += 1;
            let (p, name) = match self.map.find_entry(idx) {
                Some(EntryItem(p, n))       => (p, n.name()),
                Some(EntryForeignItem(p, n))=> (p, n.name()),
                Some(EntryTraitItem(p, n))  => (p, n.name()),
                Some(EntryImplItem(p, n))   => (p, n.name()),
                Some(EntryVariant(p, n))    => (p, n.name()),
                _ => continue,
            };
            if self.matches_names(p, name) {
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
impl Named for TraitItem {
    fn name(&self) -> Name {
        match *self {
            RequiredMethod(ref tm) => tm.ident.name,
            ProvidedMethod(ref m) => m.name(),
            TypeTraitItem(ref at) => at.ident.name,
        }
    }
}
impl Named for ImplItem {
    fn name(&self) -> Name {
        match *self {
            MethodImplItem(ref m) => m.name(),
            TypeImplItem(ref td) => td.ident.name,
        }
    }
}
impl Named for Method {
    fn name(&self) -> Name {
        match self.node {
            MethDecl(i, _, _, _, _, _, _, _) => i.name,
            MethMac(_) => fail!("encountered unexpanded method macro."),
        }
    }
}

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
    /// The node in which we are currently mapping (an item or a method).
    parent: NodeId
}

impl<'ast> NodeCollector<'ast> {
    fn insert_entry(&mut self, id: NodeId, entry: MapEntry<'ast>) {
        self.map.grow_set(id as uint, &NotPresent, entry);
        debug!("ast_map: {} => {}", id, entry);
    }

    fn insert(&mut self, id: NodeId, node: Node<'ast>) {
        let entry = MapEntry::from_node(self.parent, node);
        self.insert_entry(id, entry);
    }

    fn visit_fn_decl(&mut self, decl: &'ast FnDecl) {
        for a in decl.inputs.iter() {
            self.insert(a.id, NodeArg(&*a.pat));
        }
    }
}

impl<'ast> Visitor<'ast> for NodeCollector<'ast> {
    fn visit_item(&mut self, i: &'ast Item) {
        self.insert(i.id, NodeItem(i));
        let parent = self.parent;
        self.parent = i.id;
        match i.node {
            ItemImpl(_, _, _, ref impl_items) => {
                for impl_item in impl_items.iter() {
                    match *impl_item {
                        MethodImplItem(ref m) => {
                            self.insert(m.id, NodeImplItem(impl_item));
                        }
                        TypeImplItem(ref t) => {
                            self.insert(t.id, NodeImplItem(impl_item));
                        }
                    }
                }
            }
            ItemEnum(ref enum_definition, _) => {
                for v in enum_definition.variants.iter() {
                    self.insert(v.node.id, NodeVariant(&**v));
                }
            }
            ItemForeignMod(ref nm) => {
                for nitem in nm.items.iter() {
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
                    match *b {
                        TraitTyParamBound(ref t) => {
                            self.insert(t.ref_id, NodeItem(i));
                        }
                        _ => {}
                    }
                }

                for tm in trait_items.iter() {
                    match *tm {
                        RequiredMethod(ref m) => {
                            self.insert(m.id, NodeTraitItem(tm));
                        }
                        ProvidedMethod(ref m) => {
                            self.insert(m.id, NodeTraitItem(tm));
                        }
                        TypeTraitItem(ref typ) => {
                            self.insert(typ.id, NodeTraitItem(tm));
                        }
                    }
                }
            }
            _ => {}
        }
        visit::walk_item(self, i);
        self.parent = parent;
    }

    fn visit_pat(&mut self, pat: &'ast Pat) {
        self.insert(pat.id, match pat.node {
            // Note: this is at least *potentially* a pattern...
            PatIdent(..) => NodeLocal(pat),
            _ => NodePat(pat)
        });
        visit::walk_pat(self, pat);
    }

    fn visit_expr(&mut self, expr: &'ast Expr) {
        self.insert(expr.id, NodeExpr(expr));
        visit::walk_expr(self, expr);
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt) {
        self.insert(ast_util::stmt_id(stmt), NodeStmt(stmt));
        visit::walk_stmt(self, stmt);
    }

    fn visit_ty_method(&mut self, m: &'ast TypeMethod) {
        let parent = self.parent;
        self.parent = m.id;
        self.visit_fn_decl(&*m.decl);
        visit::walk_ty_method(self, m);
        self.parent = parent;
    }

    fn visit_fn(&mut self, fk: visit::FnKind<'ast>, fd: &'ast FnDecl,
                b: &'ast Block, s: Span, id: NodeId) {
        match fk {
            visit::FkMethod(..) => {
                let parent = self.parent;
                self.parent = id;
                self.visit_fn_decl(fd);
                visit::walk_fn(self, fk, fd, b, s);
                self.parent = parent;
            }
            _ => {
                self.visit_fn_decl(fd);
                visit::walk_fn(self, fk, fd, b, s);
            }
        }
    }

    fn visit_ty(&mut self, ty: &'ast Ty) {
        match ty.node {
            TyClosure(ref fd) | TyProc(ref fd) => {
                self.visit_fn_decl(&*fd.decl);
            }
            TyBareFn(ref fd) => {
                self.visit_fn_decl(&*fd.decl);
            }
            TyUnboxedFn(ref fd) => {
                self.visit_fn_decl(&*fd.decl);
            }
            _ => {}
        }
        visit::walk_ty(self, ty);
    }

    fn visit_block(&mut self, block: &'ast Block) {
        self.insert(block.id, NodeBlock(block));
        visit::walk_block(self, block);
    }

    fn visit_lifetime_ref(&mut self, lifetime: &'ast Lifetime) {
        self.insert(lifetime.id, NodeLifetime(lifetime));
    }

    fn visit_lifetime_decl(&mut self, def: &'ast LifetimeDef) {
        self.visit_lifetime_ref(&def.lifetime);
    }
}

pub fn map_crate<'ast, F: FoldOps>(forest: &'ast mut Forest, fold_ops: F) -> Map<'ast> {
    // Replace the crate with an empty one to take it out.
    let krate = mem::replace(&mut forest.krate, Crate {
        module: Mod {
            inner: DUMMY_SP,
            view_items: vec![],
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
        parent: CRATE_NODE_ID
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
        IITraitItem(d, ti) => match ti {
            ProvidedMethod(m) => {
                IITraitItem(fld.fold_ops.new_def_id(d),
                            ProvidedMethod(fld.fold_method(m)
                                              .expect_one("expected one method")))
            }
            RequiredMethod(ty_m) => {
                IITraitItem(fld.fold_ops.new_def_id(d),
                            RequiredMethod(fld.fold_type_method(ty_m)))
            }
            TypeTraitItem(at) => {
                IITraitItem(
                    fld.fold_ops.new_def_id(d),
                    TypeTraitItem(P(fld.fold_associated_type((*at).clone()))))
            }
        },
        IIImplItem(d, m) => match m {
            MethodImplItem(m) => {
                IIImplItem(fld.fold_ops.new_def_id(d),
                           MethodImplItem(fld.fold_method(m)
                                             .expect_one("expected one method")))
            }
            TypeImplItem(t) => {
                IIImplItem(fld.fold_ops.new_def_id(d),
                           TypeImplItem(P(fld.fold_typedef((*t).clone()))))
            }
        },
        IIForeign(i) => IIForeign(fld.fold_foreign_item(i))
    };

    let ii_parent = map.forest.inlined_items.alloc(InlinedParent {
        path: path,
        ii: ii
    });

    let mut collector = NodeCollector {
        map: mem::replace(&mut *map.map.borrow_mut(), vec![]),
        parent: fld.new_id(DUMMY_NODE_ID)
    };
    let ii_parent_id = collector.parent;
    collector.insert_entry(ii_parent_id, RootInlinedParent(ii_parent));
    visit::walk_inlined_item(&mut collector, &ii_parent.ii);

    // Methods get added to the AST map when their impl is visited.  Since we
    // don't decode and instantiate the impl, but just the method, we have to
    // add it to the table now. Likewise with foreign items.
    match ii_parent.ii {
        IIItem(_) => {}
        IITraitItem(_, ref trait_item) => {
            let trait_item_id = match *trait_item {
                ProvidedMethod(ref m) => m.id,
                RequiredMethod(ref m) => m.id,
                TypeTraitItem(ref ty) => ty.id,
            };

            collector.insert(trait_item_id, NodeTraitItem(trait_item));
        }
        IIImplItem(_, ref impl_item) => {
            let impl_item_id = match *impl_item {
                MethodImplItem(ref m) => m.id,
                TypeImplItem(ref ti) => ti.id,
            };

            collector.insert(impl_item_id, NodeImplItem(impl_item));
        }
        IIForeign(ref i) => {
            collector.insert(i.id, NodeForeignItem(&**i));
        }
    }
    *map.map.borrow_mut() = collector.map;
    &ii_parent.ii
}

pub trait NodePrinter {
    fn print_node(&mut self, node: &Node) -> IoResult<()>;
}

impl<'a> NodePrinter for pprust::State<'a> {
    fn print_node(&mut self, node: &Node) -> IoResult<()> {
        match *node {
            NodeItem(a)        => self.print_item(&*a),
            NodeForeignItem(a) => self.print_foreign_item(&*a),
            NodeTraitItem(a)   => self.print_trait_method(&*a),
            NodeImplItem(a)    => self.print_impl_item(&*a),
            NodeVariant(a)     => self.print_variant(&*a),
            NodeExpr(a)        => self.print_expr(&*a),
            NodeStmt(a)        => self.print_stmt(&*a),
            NodePat(a)         => self.print_pat(&*a),
            NodeBlock(a)       => self.print_block(&*a),
            NodeLifetime(a)    => self.print_lifetime(&*a),

            // these cases do not carry enough information in the
            // ast_map to reconstruct their full structure for pretty
            // printing.
            NodeLocal(_)       => fail!("cannot print isolated Local"),
            NodeArg(_)         => fail!("cannot print isolated Arg"),
            NodeStructCtor(_)  => fail!("cannot print isolated StructCtor"),
        }
    }
}

fn node_id_to_string(map: &Map, id: NodeId) -> String {
    match map.find(id) {
        Some(NodeItem(item)) => {
            let path_str = map.path_to_str_with_ident(id, item.ident);
            let item_str = match item.node {
                ItemStatic(..) => "static",
                ItemFn(..) => "fn",
                ItemMod(..) => "mod",
                ItemForeignMod(..) => "foreign mod",
                ItemTy(..) => "ty",
                ItemEnum(..) => "enum",
                ItemStruct(..) => "struct",
                ItemTrait(..) => "trait",
                ItemImpl(..) => "impl",
                ItemMac(..) => "macro"
            };
            format!("{} {} (id={})", item_str, path_str, id)
        }
        Some(NodeForeignItem(item)) => {
            let path_str = map.path_to_str_with_ident(id, item.ident);
            format!("foreign item {} (id={})", path_str, id)
        }
        Some(NodeImplItem(ref ii)) => {
            match **ii {
                MethodImplItem(ref m) => {
                    match m.node {
                        MethDecl(ident, _, _, _, _, _, _, _) =>
                            format!("method {} in {} (id={})",
                                    token::get_ident(ident),
                                    map.path_to_string(id), id),
                        MethMac(ref mac) =>
                            format!("method macro {} (id={})",
                                    pprust::mac_to_string(mac), id)
                    }
                }
                TypeImplItem(ref t) => {
                    format!("typedef {} in {} (id={})",
                            token::get_ident(t.ident),
                            map.path_to_string(id),
                            id)
                }
            }
        }
        Some(NodeTraitItem(ref tm)) => {
            match **tm {
                RequiredMethod(_) | ProvidedMethod(_) => {
                    let m = ast_util::trait_item_to_ty_method(&**tm);
                    format!("method {} in {} (id={})",
                            token::get_ident(m.ident),
                            map.path_to_string(id),
                            id)
                }
                TypeTraitItem(ref t) => {
                    format!("type item {} in {} (id={})",
                            token::get_ident(t.ident),
                            map.path_to_string(id),
                            id)
                }
            }
        }
        Some(NodeVariant(ref variant)) => {
            format!("variant {} in {} (id={})",
                    token::get_ident(variant.node.name),
                    map.path_to_string(id), id)
        }
        Some(NodeExpr(ref expr)) => {
            format!("expr {} (id={})", pprust::expr_to_string(&**expr), id)
        }
        Some(NodeStmt(ref stmt)) => {
            format!("stmt {} (id={})", pprust::stmt_to_string(&**stmt), id)
        }
        Some(NodeArg(ref pat)) => {
            format!("arg {} (id={})", pprust::pat_to_string(&**pat), id)
        }
        Some(NodeLocal(ref pat)) => {
            format!("local {} (id={})", pprust::pat_to_string(&**pat), id)
        }
        Some(NodePat(ref pat)) => {
            format!("pat {} (id={})", pprust::pat_to_string(&**pat), id)
        }
        Some(NodeBlock(ref block)) => {
            format!("block {} (id={})", pprust::block_to_string(&**block), id)
        }
        Some(NodeStructCtor(_)) => {
            format!("struct_ctor {} (id={})", map.path_to_string(id), id)
        }
        Some(NodeLifetime(ref l)) => {
            format!("lifetime {} (id={})",
                    pprust::lifetime_to_string(&**l), id)
        }
        None => {
            format!("unknown node (id={})", id)
        }
    }
}
