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
use codemap::Span;
use fold::Folder;
use fold;
use parse::token;
use print::pprust;
use util::small_vector::SmallVector;

use std::cell::RefCell;
use std::fmt;
use std::gc::{Gc, GC};
use std::iter;
use std::slice;

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
pub struct Values<'a, T>(pub slice::Items<'a, T>);

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

#[deriving(Clone)]
pub enum Node {
    NodeItem(Gc<Item>),
    NodeForeignItem(Gc<ForeignItem>),
    NodeTraitMethod(Gc<TraitMethod>),
    NodeMethod(Gc<Method>),
    NodeVariant(P<Variant>),
    NodeExpr(Gc<Expr>),
    NodeStmt(Gc<Stmt>),
    NodeArg(Gc<Pat>),
    NodeLocal(Gc<Pat>),
    NodePat(Gc<Pat>),
    NodeBlock(P<Block>),

    /// NodeStructCtor represents a tuple struct.
    NodeStructCtor(Gc<StructDef>),

    NodeLifetime(Gc<Lifetime>),
}

// The odd layout is to bring down the total size.
#[deriving(Clone)]
enum MapEntry {
    // Placeholder for holes in the map.
    NotPresent,

    // All the node types, with a parent ID.
    EntryItem(NodeId, Gc<Item>),
    EntryForeignItem(NodeId, Gc<ForeignItem>),
    EntryTraitMethod(NodeId, Gc<TraitMethod>),
    EntryMethod(NodeId, Gc<Method>),
    EntryVariant(NodeId, P<Variant>),
    EntryExpr(NodeId, Gc<Expr>),
    EntryStmt(NodeId, Gc<Stmt>),
    EntryArg(NodeId, Gc<Pat>),
    EntryLocal(NodeId, Gc<Pat>),
    EntryPat(NodeId, Gc<Pat>),
    EntryBlock(NodeId, P<Block>),
    EntryStructCtor(NodeId, Gc<StructDef>),
    EntryLifetime(NodeId, Gc<Lifetime>),

    // Roots for node trees.
    RootCrate,
    RootInlinedParent(P<InlinedParent>)
}

struct InlinedParent {
    path: Vec<PathElem> ,
    // Required by NodeTraitMethod and NodeMethod.
    def_id: DefId
}

impl MapEntry {
    fn parent(&self) -> Option<NodeId> {
        Some(match *self {
            EntryItem(id, _) => id,
            EntryForeignItem(id, _) => id,
            EntryTraitMethod(id, _) => id,
            EntryMethod(id, _) => id,
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

    fn to_node(&self) -> Option<Node> {
        Some(match *self {
            EntryItem(_, p) => NodeItem(p),
            EntryForeignItem(_, p) => NodeForeignItem(p),
            EntryTraitMethod(_, p) => NodeTraitMethod(p),
            EntryMethod(_, p) => NodeMethod(p),
            EntryVariant(_, p) => NodeVariant(p),
            EntryExpr(_, p) => NodeExpr(p),
            EntryStmt(_, p) => NodeStmt(p),
            EntryArg(_, p) => NodeArg(p),
            EntryLocal(_, p) => NodeLocal(p),
            EntryPat(_, p) => NodePat(p),
            EntryBlock(_, p) => NodeBlock(p),
            EntryStructCtor(_, p) => NodeStructCtor(p),
            EntryLifetime(_, p) => NodeLifetime(p),
            _ => return None
        })
    }
}

pub struct Map {
    /// NodeIds are sequential integers from 0, so we can be
    /// super-compact by storing them in a vector. Not everything with
    /// a NodeId is in the map, but empirically the occupancy is about
    /// 75-80%, so there's not too much overhead (certainly less than
    /// a hashmap, since they (at the time of writing) have a maximum
    /// of 75% occupancy).
    ///
    /// Also, indexing is pretty quick when you've got a vector and
    /// plain old integers.
    map: RefCell<Vec<MapEntry> >
}

impl Map {
    fn find_entry(&self, id: NodeId) -> Option<MapEntry> {
        let map = self.map.borrow();
        if map.len() > id as uint {
            Some(*map.get(id as uint))
        } else {
            None
        }
    }

    /// Retrieve the Node corresponding to `id`, failing if it cannot
    /// be found.
    pub fn get(&self, id: NodeId) -> Node {
        match self.find(id) {
            Some(node) => node,
            None => fail!("couldn't find node id {} in the AST map", id)
        }
    }

    /// Retrieve the Node corresponding to `id`, returning None if
    /// cannot be found.
    pub fn find(&self, id: NodeId) -> Option<Node> {
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
            Some(RootInlinedParent(data)) => data.def_id,
            _ => ast_util::local_def(parent)
        }
    }

    pub fn get_foreign_abi(&self, id: NodeId) -> abi::Abi {
        let parent = self.get_parent(id);
        let abi = match self.find_entry(parent) {
            Some(EntryItem(_, i)) => match i.node {
                ItemForeignMod(ref nm) => Some(nm.abi),
                _ => None
            },
            // Wrong but OK, because the only inlined foreign items are intrinsics.
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

    pub fn expect_item(&self, id: NodeId) -> Gc<Item> {
        match self.find(id) {
            Some(NodeItem(item)) => item,
            _ => fail!("expected item, found {}", self.node_to_string(id))
        }
    }

    pub fn expect_struct(&self, id: NodeId) -> Gc<StructDef> {
        match self.find(id) {
            Some(NodeItem(i)) => {
                match i.node {
                    ItemStruct(struct_def, _) => struct_def,
                    _ => fail!("struct ID bound to non-struct")
                }
            }
            Some(NodeVariant(ref variant)) => {
                match (*variant).node.kind {
                    StructVariantKind(struct_def) => struct_def,
                    _ => fail!("struct ID bound to enum variant that isn't struct-like"),
                }
            }
            _ => fail!(format!("expected struct, found {}", self.node_to_string(id))),
        }
    }

    pub fn expect_variant(&self, id: NodeId) -> P<Variant> {
        match self.find(id) {
            Some(NodeVariant(variant)) => variant,
            _ => fail!(format!("expected variant, found {}", self.node_to_string(id))),
        }
    }

    pub fn expect_foreign_item(&self, id: NodeId) -> Gc<ForeignItem> {
        match self.find(id) {
            Some(NodeForeignItem(item)) => item,
            _ => fail!("expected foreign item, found {}", self.node_to_string(id))
        }
    }

    pub fn get_path_elem(&self, id: NodeId) -> PathElem {
        match self.get(id) {
            NodeItem(item) => {
                match item.node {
                    ItemMod(_) | ItemForeignMod(_) => {
                        PathMod(item.ident.name)
                    }
                    _ => PathName(item.ident.name)
                }
            }
            NodeForeignItem(i) => PathName(i.ident.name),
            NodeMethod(m) => PathName(m.ident.name),
            NodeTraitMethod(tm) => match *tm {
                Required(ref m) => PathName(m.ident.name),
                Provided(ref m) => PathName(m.ident.name)
            },
            NodeVariant(v) => PathName(v.node.name.name),
            node => fail!("no path elem for {:?}", node)
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
            path_to_string(path.chain(Some(PathName(i.name)).move_iter()))
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

    pub fn with_attrs<T>(&self, id: NodeId, f: |Option<&[Attribute]>| -> T) -> T {
        let node = self.get(id);
        let attrs = match node {
            NodeItem(ref i) => Some(i.attrs.as_slice()),
            NodeForeignItem(ref fi) => Some(fi.attrs.as_slice()),
            NodeTraitMethod(ref tm) => match **tm {
                Required(ref type_m) => Some(type_m.attrs.as_slice()),
                Provided(ref m) => Some(m.attrs.as_slice())
            },
            NodeMethod(ref m) => Some(m.attrs.as_slice()),
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

    pub fn opt_span(&self, id: NodeId) -> Option<Span> {
        let sp = match self.find(id) {
            Some(NodeItem(item)) => item.span,
            Some(NodeForeignItem(foreign_item)) => foreign_item.span,
            Some(NodeTraitMethod(trait_method)) => {
                match *trait_method {
                    Required(ref type_method) => type_method.span,
                    Provided(ref method) => method.span,
                }
            }
            Some(NodeMethod(method)) => method.span,
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

pub trait FoldOps {
    fn new_id(&self, id: NodeId) -> NodeId {
        id
    }
    fn new_span(&self, span: Span) -> Span {
        span
    }
}

pub struct Ctx<'a, F> {
    map: &'a Map,
    // The node in which we are currently mapping (an item or a method).
    // When equal to DUMMY_NODE_ID, the next mapped node becomes the parent.
    parent: NodeId,
    fold_ops: F
}

impl<'a, F> Ctx<'a, F> {
    fn insert(&self, id: NodeId, entry: MapEntry) {
        (*self.map.map.borrow_mut()).grow_set(id as uint, &NotPresent, entry);
    }
}

impl<'a, F: FoldOps> Folder for Ctx<'a, F> {
    fn new_id(&mut self, id: NodeId) -> NodeId {
        let id = self.fold_ops.new_id(id);
        if self.parent == DUMMY_NODE_ID {
            self.parent = id;
        }
        id
    }

    fn new_span(&mut self, span: Span) -> Span {
        self.fold_ops.new_span(span)
    }

    fn fold_item(&mut self, i: Gc<Item>) -> SmallVector<Gc<Item>> {
        let parent = self.parent;
        self.parent = DUMMY_NODE_ID;

        let i = fold::noop_fold_item(&*i, self).expect_one("expected one item");
        assert_eq!(self.parent, i.id);

        match i.node {
            ItemImpl(_, _, _, ref ms) => {
                for &m in ms.iter() {
                    self.insert(m.id, EntryMethod(self.parent, m));
                }
            }
            ItemEnum(ref enum_definition, _) => {
                for &v in enum_definition.variants.iter() {
                    self.insert(v.node.id, EntryVariant(self.parent, v));
                }
            }
            ItemForeignMod(ref nm) => {
                for nitem in nm.items.iter() {
                    self.insert(nitem.id, EntryForeignItem(self.parent,
                                                           nitem.clone()));
                }
            }
            ItemStruct(ref struct_def, _) => {
                // If this is a tuple-like struct, register the constructor.
                match struct_def.ctor_id {
                    Some(ctor_id) => {
                        self.insert(ctor_id, EntryStructCtor(self.parent,
                                                             struct_def.clone()));
                    }
                    None => {}
                }
            }
            ItemTrait(_, _, ref traits, ref methods) => {
                for t in traits.iter() {
                    self.insert(t.ref_id, EntryItem(self.parent, i));
                }

                for tm in methods.iter() {
                    match *tm {
                        Required(ref m) => {
                            self.insert(m.id, EntryTraitMethod(self.parent,
                                                               box(GC) (*tm).clone()));
                        }
                        Provided(m) => {
                            self.insert(m.id, EntryTraitMethod(self.parent,
                                                               box(GC) Provided(m)));
                        }
                    }
                }
            }
            _ => {}
        }

        self.parent = parent;
        self.insert(i.id, EntryItem(self.parent, i));

        SmallVector::one(i)
    }

    fn fold_pat(&mut self, pat: Gc<Pat>) -> Gc<Pat> {
        let pat = fold::noop_fold_pat(pat, self);
        match pat.node {
            PatIdent(..) => {
                // Note: this is at least *potentially* a pattern...
                self.insert(pat.id, EntryLocal(self.parent, pat));
            }
            _ => {
                self.insert(pat.id, EntryPat(self.parent, pat));
            }
        }

        pat
    }

    fn fold_expr(&mut self, expr: Gc<Expr>) -> Gc<Expr> {
        let expr = fold::noop_fold_expr(expr, self);

        self.insert(expr.id, EntryExpr(self.parent, expr));

        expr
    }

    fn fold_stmt(&mut self, stmt: &Stmt) -> SmallVector<Gc<Stmt>> {
        let stmt = fold::noop_fold_stmt(stmt, self).expect_one("expected one statement");
        self.insert(ast_util::stmt_id(&*stmt), EntryStmt(self.parent, stmt));
        SmallVector::one(stmt)
    }

    fn fold_type_method(&mut self, m: &TypeMethod) -> TypeMethod {
        let parent = self.parent;
        self.parent = DUMMY_NODE_ID;
        let m = fold::noop_fold_type_method(m, self);
        assert_eq!(self.parent, m.id);
        self.parent = parent;
        m
    }

    fn fold_method(&mut self, m: Gc<Method>) -> Gc<Method> {
        let parent = self.parent;
        self.parent = DUMMY_NODE_ID;
        let m = fold::noop_fold_method(&*m, self);
        assert_eq!(self.parent, m.id);
        self.parent = parent;
        m
    }

    fn fold_fn_decl(&mut self, decl: &FnDecl) -> P<FnDecl> {
        let decl = fold::noop_fold_fn_decl(decl, self);
        for a in decl.inputs.iter() {
            self.insert(a.id, EntryArg(self.parent, a.pat));
        }
        decl
    }

    fn fold_block(&mut self, block: P<Block>) -> P<Block> {
        let block = fold::noop_fold_block(block, self);
        self.insert(block.id, EntryBlock(self.parent, block));
        block
    }

    fn fold_lifetime(&mut self, lifetime: &Lifetime) -> Lifetime {
        let lifetime = fold::noop_fold_lifetime(lifetime, self);
        self.insert(lifetime.id, EntryLifetime(self.parent, box(GC) lifetime));
        lifetime
    }
}

pub fn map_crate<F: FoldOps>(krate: Crate, fold_ops: F) -> (Crate, Map) {
    let map = Map { map: RefCell::new(Vec::new()) };
    let krate = {
        let mut cx = Ctx {
            map: &map,
            parent: CRATE_NODE_ID,
            fold_ops: fold_ops
        };
        cx.insert(CRATE_NODE_ID, RootCrate);
        cx.fold_crate(krate)
    };

    if log_enabled!(::log::DEBUG) {
        let map = map.map.borrow();
        // This only makes sense for ordered stores; note the
        // enumerate to count the number of entries.
        let (entries_less_1, _) = (*map).iter().filter(|&x| {
            match *x {
                NotPresent => false,
                _ => true
            }
        }).enumerate().last().expect("AST map was empty after folding?");

        let entries = entries_less_1 + 1;
        let vector_length = (*map).len();
        debug!("The AST map has {} entries with a maximum of {}: occupancy {:.1}%",
              entries, vector_length, (entries as f64 / vector_length as f64) * 100.);
    }

    (krate, map)
}

// Used for items loaded from external crate that are being inlined into this
// crate.  The `path` should be the path to the item but should not include
// the item itself.
pub fn map_decoded_item<F: FoldOps>(map: &Map,
                                    path: Vec<PathElem> ,
                                    fold_ops: F,
                                    fold: |&mut Ctx<F>| -> InlinedItem)
                                    -> InlinedItem {
    let mut cx = Ctx {
        map: map,
        parent: DUMMY_NODE_ID,
        fold_ops: fold_ops
    };

    // Generate a NodeId for the RootInlinedParent inserted below.
    cx.new_id(DUMMY_NODE_ID);

    // Methods get added to the AST map when their impl is visited.  Since we
    // don't decode and instantiate the impl, but just the method, we have to
    // add it to the table now. Likewise with foreign items.
    let mut def_id = DefId { krate: LOCAL_CRATE, node: DUMMY_NODE_ID };
    let ii = fold(&mut cx);
    match ii {
        IIItem(_) => {}
        IIMethod(impl_did, is_provided, m) => {
            let entry = if is_provided {
                EntryTraitMethod(cx.parent, box(GC) Provided(m))
            } else {
                EntryMethod(cx.parent, m)
            };
            cx.insert(m.id, entry);
            def_id = impl_did;
        }
        IIForeign(i) => {
            cx.insert(i.id, EntryForeignItem(cx.parent, i));
        }
    }

    cx.insert(cx.parent, RootInlinedParent(P(InlinedParent {
        path: path,
        def_id: def_id
    })));

    ii
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
        Some(NodeMethod(m)) => {
            format!("method {} in {} (id={})",
                    token::get_ident(m.ident),
                    map.path_to_string(id), id)
        }
        Some(NodeTraitMethod(ref tm)) => {
            let m = ast_util::trait_method_to_ty_method(&**tm);
            format!("method {} in {} (id={})",
                    token::get_ident(m.ident),
                    map.path_to_string(id), id)
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
