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
use ast;
use ast_util;
use codemap::Span;
use fold::Folder;
use fold;
use parse::token;
use print::pprust;
use util::small_vector::SmallVector;

use std::cell::RefCell;
use std::fmt;
use std::iter;
use std::slice;
use std::string::String;

#[deriving(Clone, PartialEq)]
pub enum PathElem {
    PathMod(ast::Name),
    PathName(ast::Name)
}

impl PathElem {
    pub fn name(&self) -> ast::Name {
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

pub fn path_to_str<PI: Iterator<PathElem>>(mut path: PI) -> String {
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
    NodeItem(@ast::Item),
    NodeForeignItem(@ast::ForeignItem),
    NodeTraitMethod(@ast::TraitMethod),
    NodeMethod(@ast::Method),
    NodeVariant(ast::P<ast::Variant>),
    NodeExpr(@ast::Expr),
    NodeStmt(@ast::Stmt),
    NodeArg(@ast::Pat),
    NodeLocal(@ast::Pat),
    NodePat(@ast::Pat),
    NodeBlock(ast::P<ast::Block>),

    /// NodeStructCtor represents a tuple struct.
    NodeStructCtor(@ast::StructDef),

    NodeLifetime(@ast::Lifetime),
}

// The odd layout is to bring down the total size.
#[deriving(Clone)]
enum MapEntry {
    // Placeholder for holes in the map.
    NotPresent,

    // All the node types, with a parent ID.
    EntryItem(ast::NodeId, @ast::Item),
    EntryForeignItem(ast::NodeId, @ast::ForeignItem),
    EntryTraitMethod(ast::NodeId, @ast::TraitMethod),
    EntryMethod(ast::NodeId, @ast::Method),
    EntryVariant(ast::NodeId, ast::P<ast::Variant>),
    EntryExpr(ast::NodeId, @ast::Expr),
    EntryStmt(ast::NodeId, @ast::Stmt),
    EntryArg(ast::NodeId, @ast::Pat),
    EntryLocal(ast::NodeId, @ast::Pat),
    EntryPat(ast::NodeId, @ast::Pat),
    EntryBlock(ast::NodeId, ast::P<ast::Block>),
    EntryStructCtor(ast::NodeId, @ast::StructDef),
    EntryLifetime(ast::NodeId, @ast::Lifetime),

    // Roots for node trees.
    RootCrate,
    RootInlinedParent(ast::P<InlinedParent>)
}

struct InlinedParent {
    path: Vec<PathElem> ,
    // Required by NodeTraitMethod and NodeMethod.
    def_id: ast::DefId
}

impl MapEntry {
    fn parent(&self) -> Option<ast::NodeId> {
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
    fn find_entry(&self, id: ast::NodeId) -> Option<MapEntry> {
        let map = self.map.borrow();
        if map.len() > id as uint {
            Some(*map.get(id as uint))
        } else {
            None
        }
    }

    /// Retrieve the Node corresponding to `id`, failing if it cannot
    /// be found.
    pub fn get(&self, id: ast::NodeId) -> Node {
        match self.find(id) {
            Some(node) => node,
            None => fail!("couldn't find node id {} in the AST map", id)
        }
    }

    /// Retrieve the Node corresponding to `id`, returning None if
    /// cannot be found.
    pub fn find(&self, id: ast::NodeId) -> Option<Node> {
        self.find_entry(id).and_then(|x| x.to_node())
    }

    /// Retrieve the parent NodeId for `id`, or `id` itself if no
    /// parent is registered in this map.
    pub fn get_parent(&self, id: ast::NodeId) -> ast::NodeId {
        self.find_entry(id).and_then(|x| x.parent()).unwrap_or(id)
    }

    pub fn get_parent_did(&self, id: ast::NodeId) -> ast::DefId {
        let parent = self.get_parent(id);
        match self.find_entry(parent) {
            Some(RootInlinedParent(data)) => data.def_id,
            _ => ast_util::local_def(parent)
        }
    }

    pub fn get_foreign_abi(&self, id: ast::NodeId) -> abi::Abi {
        let parent = self.get_parent(id);
        let abi = match self.find_entry(parent) {
            Some(EntryItem(_, i)) => match i.node {
                ast::ItemForeignMod(ref nm) => Some(nm.abi),
                _ => None
            },
            // Wrong but OK, because the only inlined foreign items are intrinsics.
            Some(RootInlinedParent(_)) => Some(abi::RustIntrinsic),
            _ => None
        };
        match abi {
            Some(abi) => abi,
            None => fail!("expected foreign mod or inlined parent, found {}",
                          self.node_to_str(parent))
        }
    }

    pub fn get_foreign_vis(&self, id: ast::NodeId) -> ast::Visibility {
        let vis = self.expect_foreign_item(id).vis;
        match self.find(self.get_parent(id)) {
            Some(NodeItem(i)) => vis.inherit_from(i.vis),
            _ => vis
        }
    }

    pub fn expect_item(&self, id: ast::NodeId) -> @ast::Item {
        match self.find(id) {
            Some(NodeItem(item)) => item,
            _ => fail!("expected item, found {}", self.node_to_str(id))
        }
    }

    pub fn expect_struct(&self, id: ast::NodeId) -> @ast::StructDef {
        match self.find(id) {
            Some(NodeItem(i)) => {
                match i.node {
                    ast::ItemStruct(struct_def, _) => struct_def,
                    _ => fail!("struct ID bound to non-struct")
                }
            }
            Some(NodeVariant(ref variant)) => {
                match (*variant).node.kind {
                    ast::StructVariantKind(struct_def) => struct_def,
                    _ => fail!("struct ID bound to enum variant that isn't struct-like"),
                }
            }
            _ => fail!(format!("expected struct, found {}", self.node_to_str(id))),
        }
    }

    pub fn expect_variant(&self, id: ast::NodeId) -> ast::P<ast::Variant> {
        match self.find(id) {
            Some(NodeVariant(variant)) => variant,
            _ => fail!(format!("expected variant, found {}", self.node_to_str(id))),
        }
    }

    pub fn expect_foreign_item(&self, id: ast::NodeId) -> @ast::ForeignItem {
        match self.find(id) {
            Some(NodeForeignItem(item)) => item,
            _ => fail!("expected foreign item, found {}", self.node_to_str(id))
        }
    }

    pub fn get_path_elem(&self, id: ast::NodeId) -> PathElem {
        match self.get(id) {
            NodeItem(item) => {
                match item.node {
                    ast::ItemMod(_) | ast::ItemForeignMod(_) => {
                        PathMod(item.ident.name)
                    }
                    _ => PathName(item.ident.name)
                }
            }
            NodeForeignItem(i) => PathName(i.ident.name),
            NodeMethod(m) => PathName(m.ident.name),
            NodeTraitMethod(tm) => match *tm {
                ast::Required(ref m) => PathName(m.ident.name),
                ast::Provided(ref m) => PathName(m.ident.name)
            },
            NodeVariant(v) => PathName(v.node.name.name),
            node => fail!("no path elem for {:?}", node)
        }
    }

    pub fn with_path<T>(&self, id: ast::NodeId, f: |PathElems| -> T) -> T {
        self.with_path_next(id, None, f)
    }

    pub fn path_to_str(&self, id: ast::NodeId) -> String {
        self.with_path(id, |path| path_to_str(path))
    }

    fn path_to_str_with_ident(&self, id: ast::NodeId, i: ast::Ident) -> String {
        self.with_path(id, |path| {
            path_to_str(path.chain(Some(PathName(i.name)).move_iter()))
        })
    }

    fn with_path_next<T>(&self, id: ast::NodeId, next: LinkedPath, f: |PathElems| -> T) -> T {
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

    pub fn with_attrs<T>(&self, id: ast::NodeId, f: |Option<&[ast::Attribute]>| -> T) -> T {
        let node = self.get(id);
        let attrs = match node {
            NodeItem(ref i) => Some(i.attrs.as_slice()),
            NodeForeignItem(ref fi) => Some(fi.attrs.as_slice()),
            NodeTraitMethod(ref tm) => match **tm {
                ast::Required(ref type_m) => Some(type_m.attrs.as_slice()),
                ast::Provided(ref m) => Some(m.attrs.as_slice())
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

    pub fn opt_span(&self, id: ast::NodeId) -> Option<Span> {
        let sp = match self.find(id) {
            Some(NodeItem(item)) => item.span,
            Some(NodeForeignItem(foreign_item)) => foreign_item.span,
            Some(NodeTraitMethod(trait_method)) => {
                match *trait_method {
                    ast::Required(ref type_method) => type_method.span,
                    ast::Provided(ref method) => method.span,
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

    pub fn span(&self, id: ast::NodeId) -> Span {
        self.opt_span(id)
            .unwrap_or_else(|| fail!("AstMap.span: could not find span for id {}", id))
    }

    pub fn node_to_str(&self, id: ast::NodeId) -> String {
        node_id_to_str(self, id)
    }
}

pub trait FoldOps {
    fn new_id(&self, id: ast::NodeId) -> ast::NodeId {
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
    parent: ast::NodeId,
    fold_ops: F
}

impl<'a, F> Ctx<'a, F> {
    fn insert(&self, id: ast::NodeId, entry: MapEntry) {
        (*self.map.map.borrow_mut()).grow_set(id as uint, &NotPresent, entry);
    }
}

impl<'a, F: FoldOps> Folder for Ctx<'a, F> {
    fn new_id(&mut self, id: ast::NodeId) -> ast::NodeId {
        let id = self.fold_ops.new_id(id);
        if self.parent == ast::DUMMY_NODE_ID {
            self.parent = id;
        }
        id
    }

    fn new_span(&mut self, span: Span) -> Span {
        self.fold_ops.new_span(span)
    }

    fn fold_item(&mut self, i: @ast::Item) -> SmallVector<@ast::Item> {
        let parent = self.parent;
        self.parent = ast::DUMMY_NODE_ID;

        let i = fold::noop_fold_item(i, self).expect_one("expected one item");
        assert_eq!(self.parent, i.id);

        match i.node {
            ast::ItemImpl(_, _, _, ref ms) => {
                for &m in ms.iter() {
                    self.insert(m.id, EntryMethod(self.parent, m));
                }
            }
            ast::ItemEnum(ref enum_definition, _) => {
                for &v in enum_definition.variants.iter() {
                    self.insert(v.node.id, EntryVariant(self.parent, v));
                }
            }
            ast::ItemForeignMod(ref nm) => {
                for &nitem in nm.items.iter() {
                    self.insert(nitem.id, EntryForeignItem(self.parent, nitem));
                }
            }
            ast::ItemStruct(struct_def, _) => {
                // If this is a tuple-like struct, register the constructor.
                match struct_def.ctor_id {
                    Some(ctor_id) => {
                        self.insert(ctor_id, EntryStructCtor(self.parent,
                                                             struct_def));
                    }
                    None => {}
                }
            }
            ast::ItemTrait(_, _, ref traits, ref methods) => {
                for t in traits.iter() {
                    self.insert(t.ref_id, EntryItem(self.parent, i));
                }

                for tm in methods.iter() {
                    match *tm {
                        ast::Required(ref m) => {
                            self.insert(m.id, EntryTraitMethod(self.parent,
                                                               @(*tm).clone()));
                        }
                        ast::Provided(m) => {
                            self.insert(m.id, EntryTraitMethod(self.parent,
                                                               @ast::Provided(m)));
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

    fn fold_pat(&mut self, pat: @ast::Pat) -> @ast::Pat {
        let pat = fold::noop_fold_pat(pat, self);
        match pat.node {
            ast::PatIdent(..) => {
                // Note: this is at least *potentially* a pattern...
                self.insert(pat.id, EntryLocal(self.parent, pat));
            }
            _ => {
                self.insert(pat.id, EntryPat(self.parent, pat));
            }
        }

        pat
    }

    fn fold_expr(&mut self, expr: @ast::Expr) -> @ast::Expr {
        let expr = fold::noop_fold_expr(expr, self);

        self.insert(expr.id, EntryExpr(self.parent, expr));

        expr
    }

    fn fold_stmt(&mut self, stmt: &ast::Stmt) -> SmallVector<@ast::Stmt> {
        let stmt = fold::noop_fold_stmt(stmt, self).expect_one("expected one statement");
        self.insert(ast_util::stmt_id(stmt), EntryStmt(self.parent, stmt));
        SmallVector::one(stmt)
    }

    fn fold_type_method(&mut self, m: &ast::TypeMethod) -> ast::TypeMethod {
        let parent = self.parent;
        self.parent = ast::DUMMY_NODE_ID;
        let m = fold::noop_fold_type_method(m, self);
        assert_eq!(self.parent, m.id);
        self.parent = parent;
        m
    }

    fn fold_method(&mut self, m: @ast::Method) -> @ast::Method {
        let parent = self.parent;
        self.parent = ast::DUMMY_NODE_ID;
        let m = fold::noop_fold_method(m, self);
        assert_eq!(self.parent, m.id);
        self.parent = parent;
        m
    }

    fn fold_fn_decl(&mut self, decl: &ast::FnDecl) -> ast::P<ast::FnDecl> {
        let decl = fold::noop_fold_fn_decl(decl, self);
        for a in decl.inputs.iter() {
            self.insert(a.id, EntryArg(self.parent, a.pat));
        }
        decl
    }

    fn fold_block(&mut self, block: ast::P<ast::Block>) -> ast::P<ast::Block> {
        let block = fold::noop_fold_block(block, self);
        self.insert(block.id, EntryBlock(self.parent, block));
        block
    }

    fn fold_lifetime(&mut self, lifetime: &ast::Lifetime) -> ast::Lifetime {
        let lifetime = fold::noop_fold_lifetime(lifetime, self);
        self.insert(lifetime.id, EntryLifetime(self.parent, @lifetime));
        lifetime
    }
}

pub fn map_crate<F: FoldOps>(krate: ast::Crate, fold_ops: F) -> (ast::Crate, Map) {
    let map = Map { map: RefCell::new(Vec::new()) };
    let krate = {
        let mut cx = Ctx {
            map: &map,
            parent: ast::CRATE_NODE_ID,
            fold_ops: fold_ops
        };
        cx.insert(ast::CRATE_NODE_ID, RootCrate);
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
                                    fold: |&mut Ctx<F>| -> ast::InlinedItem)
                                    -> ast::InlinedItem {
    let mut cx = Ctx {
        map: map,
        parent: ast::DUMMY_NODE_ID,
        fold_ops: fold_ops
    };

    // Generate a NodeId for the RootInlinedParent inserted below.
    cx.new_id(ast::DUMMY_NODE_ID);

    // Methods get added to the AST map when their impl is visited.  Since we
    // don't decode and instantiate the impl, but just the method, we have to
    // add it to the table now. Likewise with foreign items.
    let mut def_id = ast::DefId { krate: ast::LOCAL_CRATE, node: ast::DUMMY_NODE_ID };
    let ii = fold(&mut cx);
    match ii {
        ast::IIItem(_) => {}
        ast::IIMethod(impl_did, is_provided, m) => {
            let entry = if is_provided {
                EntryTraitMethod(cx.parent, @ast::Provided(m))
            } else {
                EntryMethod(cx.parent, m)
            };
            cx.insert(m.id, entry);
            def_id = impl_did;
        }
        ast::IIForeign(i) => {
            cx.insert(i.id, EntryForeignItem(cx.parent, i));
        }
    }

    cx.insert(cx.parent, RootInlinedParent(ast::P(InlinedParent {
        path: path,
        def_id: def_id
    })));

    ii
}

fn node_id_to_str(map: &Map, id: ast::NodeId) -> String {
    match map.find(id) {
        Some(NodeItem(item)) => {
            let path_str = map.path_to_str_with_ident(id, item.ident);
            let item_str = match item.node {
                ast::ItemStatic(..) => "static",
                ast::ItemFn(..) => "fn",
                ast::ItemMod(..) => "mod",
                ast::ItemForeignMod(..) => "foreign mod",
                ast::ItemTy(..) => "ty",
                ast::ItemEnum(..) => "enum",
                ast::ItemStruct(..) => "struct",
                ast::ItemTrait(..) => "trait",
                ast::ItemImpl(..) => "impl",
                ast::ItemMac(..) => "macro"
            };
            (format!("{} {} (id={})", item_str, path_str, id)).to_string()
        }
        Some(NodeForeignItem(item)) => {
            let path_str = map.path_to_str_with_ident(id, item.ident);
            (format!("foreign item {} (id={})", path_str, id)).to_string()
        }
        Some(NodeMethod(m)) => {
            (format!("method {} in {} (id={})",
                    token::get_ident(m.ident),
                    map.path_to_str(id), id)).to_string()
        }
        Some(NodeTraitMethod(ref tm)) => {
            let m = ast_util::trait_method_to_ty_method(&**tm);
            (format!("method {} in {} (id={})",
                    token::get_ident(m.ident),
                    map.path_to_str(id), id)).to_string()
        }
        Some(NodeVariant(ref variant)) => {
            (format!("variant {} in {} (id={})",
                    token::get_ident(variant.node.name),
                    map.path_to_str(id), id)).to_string()
        }
        Some(NodeExpr(expr)) => {
            (format!("expr {} (id={})",
                    pprust::expr_to_str(expr), id)).to_string()
        }
        Some(NodeStmt(stmt)) => {
            (format!("stmt {} (id={})",
                    pprust::stmt_to_str(stmt), id)).to_string()
        }
        Some(NodeArg(pat)) => {
            (format!("arg {} (id={})",
                    pprust::pat_to_str(pat), id)).to_string()
        }
        Some(NodeLocal(pat)) => {
            (format!("local {} (id={})",
                    pprust::pat_to_str(pat), id)).to_string()
        }
        Some(NodePat(pat)) => {
            (format!("pat {} (id={})", pprust::pat_to_str(pat), id)).to_string()
        }
        Some(NodeBlock(block)) => {
            (format!("block {} (id={})",
                    pprust::block_to_str(block), id)).to_string()
        }
        Some(NodeStructCtor(_)) => {
            (format!("struct_ctor {} (id={})",
                    map.path_to_str(id), id)).to_string()
        }
        Some(NodeLifetime(ref l)) => {
            (format!("lifetime {} (id={})",
                    pprust::lifetime_to_str(*l), id)).to_string()
        }
        None => {
            (format!("unknown node (id={})", id)).to_string()
        }
    }
}
