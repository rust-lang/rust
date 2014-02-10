// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::AbiSet;
use ast::*;
use ast;
use ast_util;
use codemap::Span;
use diagnostic::SpanHandler;
use fold::Folder;
use fold;
use parse::token::{get_ident_interner, IdentInterner};
use print::pprust;
use util::small_vector::SmallVector;

use std::logging;
use std::cell::RefCell;
use collections::SmallIntMap;

#[deriving(Clone, Eq)]
pub enum PathElem {
    PathMod(Ident),
    PathName(Ident),

    // A pretty name can come from an `impl` block. We attempt to select a
    // reasonable name for debuggers to see, but to guarantee uniqueness with
    // other paths the hash should also be taken into account during symbol
    // generation.
    PathPrettyName(Ident, u64),
}

impl PathElem {
    pub fn ident(&self) -> Ident {
        match *self {
            PathMod(ident)            |
            PathName(ident)           |
            PathPrettyName(ident, _) => ident
        }
    }
}

pub type Path = ~[PathElem];

pub fn path_to_str_with_sep(p: &[PathElem], sep: &str, itr: @IdentInterner)
                            -> ~str {
    let strs = p.map(|e| {
        match *e {
            PathMod(s) | PathName(s) | PathPrettyName(s, _) => {
                itr.get(s.name)
            }
        }
    });
    strs.connect(sep)
}

pub fn path_ident_to_str(p: &Path, i: Ident, itr: @IdentInterner) -> ~str {
    if p.is_empty() {
        itr.get(i.name).into_owned()
    } else {
        let string = itr.get(i.name);
        format!("{}::{}", path_to_str(*p, itr), string.as_slice())
    }
}

pub fn path_to_str(p: &[PathElem], itr: @IdentInterner) -> ~str {
    path_to_str_with_sep(p, "::", itr)
}

pub fn path_elem_to_str(pe: PathElem, itr: @IdentInterner) -> ~str {
    match pe {
        PathMod(s) | PathName(s) | PathPrettyName(s, _) => {
            itr.get(s.name).into_owned()
        }
    }
}

/// write a "pretty" version of `ty` to `out`. This is designed so
/// that symbols of `impl`'d methods give some hint of where they came
/// from, even if it's hard to read (previously they would all just be
/// listed as `__extensions__::method_name::hash`, with no indication
/// of the type).
// FIXME: these dollar signs and the names in general are actually a
//      relic of $ being one of the very few valid symbol names on
//      unix. These kinds of details shouldn't be exposed way up here
//      in the ast.
fn pretty_ty(ty: &Ty, itr: @IdentInterner, out: &mut ~str) {
    let (prefix, subty) = match ty.node {
        TyUniq(ty) => ("$UP$", &*ty),
        TyBox(ty) => ("$SP$", &*ty),
        TyPtr(MutTy { ty, mutbl }) => (if mutbl == MutMutable {"$RPmut$"} else {"$RP$"},
                                       &*ty),
        TyRptr(_, MutTy { ty, mutbl }) => (if mutbl == MutMutable {"$BPmut$"} else {"$BP$"},
                                           &*ty),

        TyVec(ty) => ("$VEC$", &*ty),
        TyFixedLengthVec(ty, _) => ("$FIXEDVEC$", &*ty),

        // these can't be represented as <prefix><contained ty>, so
        // need custom handling.
        TyNil => { out.push_str("$NIL$"); return }
        TyPath(ref path, _, _) => {
            out.push_str(itr.get(path.segments
                                     .last()
                                     .unwrap()
                                     .identifier
                                     .name).as_slice());
            return
        }
        TyTup(ref tys) => {
            out.push_str(format!("$TUP_{}$", tys.len()));
            for subty in tys.iter() {
                pretty_ty(*subty, itr, out);
                out.push_char('$');
            }
            return
        }

        // meh, better than nothing.
        TyBot => { out.push_str("$BOT$"); return }
        TyClosure(..) => { out.push_str("$CLOSURE$"); return }
        TyBareFn(..) => { out.push_str("$FN$"); return }
        TyTypeof(..) => { out.push_str("$TYPEOF$"); return }
        TyInfer(..) => { out.push_str("$INFER$"); return }

    };

    out.push_str(prefix);
    pretty_ty(subty, itr, out);
}

pub fn impl_pretty_name(trait_ref: &Option<TraitRef>, ty: &Ty) -> PathElem {
    let itr = get_ident_interner();

    let hash = (trait_ref, ty).hash();
    let mut pretty;
    match *trait_ref {
        None => pretty = ~"",
        Some(ref trait_ref) => {
            pretty = itr.get(trait_ref.path.segments.last().unwrap().identifier.name)
                        .into_owned();
            pretty.push_char('$');
        }
    };
    pretty_ty(ty, itr, &mut pretty);

    PathPrettyName(Ident::new(itr.gensym(pretty)), hash)
}

#[deriving(Clone)]
pub enum Node {
    NodeItem(@Item, @Path),
    NodeForeignItem(@ForeignItem, AbiSet, Visibility, @Path),
    NodeTraitMethod(@TraitMethod, DefId /* trait did */,
                    @Path /* path to the trait */),
    NodeMethod(@Method, DefId /* impl did */, @Path /* path to the impl */),

    /// NodeVariant represents a variant of an enum, e.g., for
    /// `enum A { B, C, D }`, there would be a NodeItem for `A`, and a
    /// NodeVariant item for each of `B`, `C`, and `D`.
    NodeVariant(P<Variant>, @Item, @Path),
    NodeExpr(@Expr),
    NodeStmt(@Stmt),
    NodeArg(@Pat),
    NodeLocal(@Pat),
    NodeBlock(P<Block>),

    /// NodeStructCtor represents a tuple struct.
    NodeStructCtor(@StructDef, @Item, @Path),
    NodeCalleeScope(@Expr)
}

impl Node {
    pub fn with_attrs<T>(&self, f: |Option<&[Attribute]>| -> T) -> T {
        let attrs = match *self {
            NodeItem(i, _) => Some(i.attrs.as_slice()),
            NodeForeignItem(fi, _, _, _) => Some(fi.attrs.as_slice()),
            NodeTraitMethod(tm, _, _) => match *tm {
                Required(ref type_m) => Some(type_m.attrs.as_slice()),
                Provided(m) => Some(m.attrs.as_slice())
            },
            NodeMethod(m, _, _) => Some(m.attrs.as_slice()),
            NodeVariant(ref v, _, _) => Some(v.node.attrs.as_slice()),
            // unit/tuple structs take the attributes straight from
            // the struct definition.
            NodeStructCtor(_, strct, _) => Some(strct.attrs.as_slice()),
            _ => None
        };
        f(attrs)
    }
}

pub struct Map {
    /// NodeIds are sequential integers from 0, so we can be
    /// super-compact by storing them in a vector. Not everything with
    /// a NodeId is in the map, but empirically the occupancy is about
    /// 75-80%, so there's not too much overhead (certainly less than
    /// a hashmap, since they (at the time of writing) have a maximum
    /// of 75% occupancy). (The additional overhead of the Option<>
    /// inside the SmallIntMap could be removed by adding an extra
    /// empty variant to Node and storing a vector here, but that was
    /// found to not make much difference.)
    ///
    /// Also, indexing is pretty quick when you've got a vector and
    /// plain old integers.
    priv map: @RefCell<SmallIntMap<Node>>
}

impl Map {
    /// Retrieve the Node corresponding to `id`, failing if it cannot
    /// be found.
    pub fn get(&self, id: ast::NodeId) -> Node {
        let map = self.map.borrow();
        *map.get().get(&(id as uint))
    }
    /// Retrieve the Node corresponding to `id`, returning None if
    /// cannot be found.
    pub fn find(&self, id: ast::NodeId) -> Option<Node> {
        let map = self.map.borrow();
        map.get().find(&(id as uint)).map(|&n| n)
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

pub struct Ctx<F> {
    map: Map,
    path: Path,
    diag: @SpanHandler,
    fold_ops: F
}

impl<F> Ctx<F> {
    fn insert(&self, id: ast::NodeId, node: Node) {
        let mut map = self.map.map.borrow_mut();
        map.get().insert(id as uint, node);
    }
}

impl<F: FoldOps> Folder for Ctx<F> {
    fn new_id(&mut self, id: ast::NodeId) -> ast::NodeId {
        self.fold_ops.new_id(id)
    }

    fn new_span(&mut self, span: Span) -> Span {
        self.fold_ops.new_span(span)
    }

    fn fold_item(&mut self, i: @Item) -> SmallVector<@Item> {
        // clone is FIXME #2543
        let item_path = @self.path.clone();
        self.path.push(match i.node {
            ItemImpl(_, ref maybe_trait, ty, _) => {
                // Right now the ident on impls is __extensions__ which isn't
                // very pretty when debugging, so attempt to select a better
                // name to use.
                impl_pretty_name(maybe_trait, ty)
            }
            ItemMod(_) | ItemForeignMod(_) => PathMod(i.ident),
            _ => PathName(i.ident)
        });

        let i = fold::noop_fold_item(i, self).expect_one("expected one item");
        self.insert(i.id, NodeItem(i, item_path));

        match i.node {
            ItemImpl(_, _, _, ref ms) => {
                // clone is FIXME #2543
                let p = @self.path.clone();
                let impl_did = ast_util::local_def(i.id);
                for &m in ms.iter() {
                    self.insert(m.id, NodeMethod(m, impl_did, p));
                }

            }
            ItemEnum(ref enum_definition, _) => {
                // clone is FIXME #2543
                let p = @self.path.clone();
                for &v in enum_definition.variants.iter() {
                    self.insert(v.node.id, NodeVariant(v, i, p));
                }
            }
            ItemForeignMod(ref nm) => {
                for nitem in nm.items.iter() {
                    // Compute the visibility for this native item.
                    let visibility = nitem.vis.inherit_from(i.vis);

                    self.insert(nitem.id,
                                // Anonymous extern mods go in the parent scope.
                                NodeForeignItem(*nitem, nm.abis, visibility, item_path));
                }
            }
            ItemStruct(struct_def, _) => {
                // If this is a tuple-like struct, register the constructor.
                match struct_def.ctor_id {
                    None => {}
                    Some(ctor_id) => {
                        // clone is FIXME #2543
                        let p = @self.path.clone();
                        self.insert(ctor_id, NodeStructCtor(struct_def, i, p));
                    }
                }
            }
            ItemTrait(_, ref traits, ref methods) => {
                for t in traits.iter() {
                    self.insert(t.ref_id, NodeItem(i, item_path));
                }

                // clone is FIXME #2543
                let p = @self.path.clone();
                for tm in methods.iter() {
                    let d_id = ast_util::local_def(i.id);
                    match *tm {
                        Required(ref m) => {
                            self.insert(m.id, NodeTraitMethod(@(*tm).clone(), d_id, p));
                        }
                        Provided(m) => {
                            self.insert(m.id, NodeTraitMethod(@Provided(m), d_id, p));
                        }
                    }
                }
            }
            _ => {}
        }

        self.path.pop().unwrap();

        SmallVector::one(i)
    }

    fn fold_pat(&mut self, pat: @Pat) -> @Pat {
        let pat = fold::noop_fold_pat(pat, self);
        match pat.node {
            PatIdent(..) => {
                // Note: this is at least *potentially* a pattern...
                self.insert(pat.id, NodeLocal(pat));
            }
            _ => {}
        }

        pat
    }

    fn fold_expr(&mut self, expr: @Expr) -> @Expr {
        let expr = fold::noop_fold_expr(expr, self);

        self.insert(expr.id, NodeExpr(expr));

        // Expressions which are or might be calls:
        {
            let r = expr.get_callee_id();
            for callee_id in r.iter() {
                self.insert(*callee_id, NodeCalleeScope(expr));
            }
        }

        expr
    }

    fn fold_stmt(&mut self, stmt: &Stmt) -> SmallVector<@Stmt> {
        let stmt = fold::noop_fold_stmt(stmt, self).expect_one("expected one statement");
        self.insert(ast_util::stmt_id(stmt), NodeStmt(stmt));
        SmallVector::one(stmt)
    }

    fn fold_method(&mut self, m: @Method) -> @Method {
        self.path.push(PathName(m.ident));
        let m = fold::noop_fold_method(m, self);
        self.path.pop();
        m
    }

    fn fold_fn_decl(&mut self, decl: &FnDecl) -> P<FnDecl> {
        let decl = fold::noop_fold_fn_decl(decl, self);
        for a in decl.inputs.iter() {
            self.insert(a.id, NodeArg(a.pat));
        }
        decl
    }

    fn fold_block(&mut self, block: P<Block>) -> P<Block> {
        let block = fold::noop_fold_block(block, self);
        self.insert(block.id, NodeBlock(block));
        block
    }
}

pub fn map_crate<F: 'static + FoldOps>(diag: @SpanHandler, c: Crate,
                                       fold_ops: F) -> (Crate, Map) {
    let mut cx = Ctx {
        map: Map { map: @RefCell::new(SmallIntMap::new()) },
        path: ~[],
        diag: diag,
        fold_ops: fold_ops
    };
    let crate = cx.fold_crate(c);

    if log_enabled!(logging::DEBUG) {
        let map = cx.map.map.borrow();
        // this only makes sense for ordered stores; note the
        // enumerate to count the number of entries.
        let (entries_less_1, (largest_id, _)) =
            map.get().iter().enumerate().last().expect("AST map was empty after folding?");

        let entries = entries_less_1 + 1;
        let vector_length = largest_id + 1;
        debug!("The AST map has {} entries with a maximum of {}: occupancy {:.1}%",
              entries, vector_length, (entries as f64 / vector_length as f64) * 100.);
    }

    (crate, cx.map)
}

// Used for items loaded from external crate that are being inlined into this
// crate.  The `path` should be the path to the item but should not include
// the item itself.
pub fn map_decoded_item<F: 'static + FoldOps>(diag: @SpanHandler,
                                              map: Map,
                                              path: Path,
                                              fold_ops: F,
                                              fold_ii: |&mut Ctx<F>| -> InlinedItem)
                                              -> InlinedItem {
    // I believe it is ok for the local IDs of inlined items from other crates
    // to overlap with the local ids from this crate, so just generate the ids
    // starting from 0.
    let mut cx = Ctx {
        map: map,
        path: path.clone(),
        diag: diag,
        fold_ops: fold_ops
    };

    let ii = fold_ii(&mut cx);

    // Methods get added to the AST map when their impl is visited.  Since we
    // don't decode and instantiate the impl, but just the method, we have to
    // add it to the table now. Likewise with foreign items.
    match ii {
        IIItem(..) => {} // fallthrough
        IIForeign(i) => {
            cx.insert(i.id, NodeForeignItem(i,
                                            AbiSet::Intrinsic(),
                                            i.vis,    // Wrong but OK
                                            @path));
        }
        IIMethod(impl_did, is_provided, m) => {
            let entry = if is_provided {
                NodeTraitMethod(@Provided(m), impl_did, @path)
            } else {
                NodeMethod(m, impl_did, @path)
            };
            cx.insert(m.id, entry);
        }
    }

    ii
}

pub fn node_id_to_str(map: Map, id: NodeId, itr: @IdentInterner) -> ~str {
    match map.find(id) {
      None => {
        format!("unknown node (id={})", id)
      }
      Some(NodeItem(item, path)) => {
        let path_str = path_ident_to_str(path, item.ident, itr);
        let item_str = match item.node {
            ItemStatic(..) => ~"static",
            ItemFn(..) => ~"fn",
            ItemMod(..) => ~"mod",
            ItemForeignMod(..) => ~"foreign mod",
            ItemTy(..) => ~"ty",
            ItemEnum(..) => ~"enum",
            ItemStruct(..) => ~"struct",
            ItemTrait(..) => ~"trait",
            ItemImpl(..) => ~"impl",
            ItemMac(..) => ~"macro"
        };
        format!("{} {} (id={})", item_str, path_str, id)
      }
      Some(NodeForeignItem(item, abi, _, path)) => {
        format!("foreign item {} with abi {:?} (id={})",
             path_ident_to_str(path, item.ident, itr), abi, id)
      }
      Some(NodeMethod(m, _, path)) => {
        let name = itr.get(m.ident.name);
        format!("method {} in {} (id={})",
             name.as_slice(), path_to_str(*path, itr), id)
      }
      Some(NodeTraitMethod(ref tm, _, path)) => {
        let m = ast_util::trait_method_to_ty_method(&**tm);
        let name = itr.get(m.ident.name);
        format!("method {} in {} (id={})",
             name.as_slice(), path_to_str(*path, itr), id)
      }
      Some(NodeVariant(ref variant, _, path)) => {
        let name = itr.get(variant.node.name.name);
        format!("variant {} in {} (id={})",
             name.as_slice(),
             path_to_str(*path, itr), id)
      }
      Some(NodeExpr(expr)) => {
        format!("expr {} (id={})", pprust::expr_to_str(expr, itr), id)
      }
      Some(NodeCalleeScope(expr)) => {
        format!("callee_scope {} (id={})", pprust::expr_to_str(expr, itr), id)
      }
      Some(NodeStmt(stmt)) => {
        format!("stmt {} (id={})",
             pprust::stmt_to_str(stmt, itr), id)
      }
      Some(NodeArg(pat)) => {
        format!("arg {} (id={})", pprust::pat_to_str(pat, itr), id)
      }
      Some(NodeLocal(pat)) => {
        format!("local {} (id={})", pprust::pat_to_str(pat, itr), id)
      }
      Some(NodeBlock(block)) => {
        format!("block {} (id={})", pprust::block_to_str(block, itr), id)
      }
      Some(NodeStructCtor(_, _, path)) => {
        format!("struct_ctor {} (id={})", path_to_str(*path, itr), id)
      }
    }
}

pub fn node_item_query<Result>(items: Map, id: NodeId, query: |@Item| -> Result, error_msg: ~str)
                       -> Result {
    match items.find(id) {
        Some(NodeItem(it, _)) => query(it),
        _ => fail!("{}", error_msg)
    }
}

pub fn node_span(items: Map, id: ast::NodeId) -> Span {
    match items.find(id) {
        Some(NodeItem(item, _)) => item.span,
        Some(NodeForeignItem(foreign_item, _, _, _)) => foreign_item.span,
        Some(NodeTraitMethod(trait_method, _, _)) => {
            match *trait_method {
                Required(ref type_method) => type_method.span,
                Provided(ref method) => method.span,
            }
        }
        Some(NodeMethod(method, _, _)) => method.span,
        Some(NodeVariant(variant, _, _)) => variant.span,
        Some(NodeExpr(expr)) => expr.span,
        Some(NodeStmt(stmt)) => stmt.span,
        Some(NodeArg(pat)) | Some(NodeLocal(pat)) => pat.span,
        Some(NodeBlock(block)) => block.span,
        Some(NodeStructCtor(_, item, _)) => item.span,
        Some(NodeCalleeScope(expr)) => expr.span,
        None => fail!("node_span: could not find id {}", id),
    }
}
