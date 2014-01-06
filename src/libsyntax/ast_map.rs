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
use fold::ast_fold;
use fold;
use parse::token::get_ident_interner;
use parse::token::ident_interner;
use parse::token::special_idents;
use print::pprust;
use util::small_vector::SmallVector;

use std::cell::RefCell;
use std::hashmap::HashMap;

#[deriving(Clone, Eq)]
pub enum path_elt {
    path_mod(Ident),
    path_name(Ident),

    // A pretty name can come from an `impl` block. We attempt to select a
    // reasonable name for debuggers to see, but to guarantee uniqueness with
    // other paths the hash should also be taken into account during symbol
    // generation.
    path_pretty_name(Ident, u64),
}

impl path_elt {
    pub fn ident(&self) -> Ident {
        match *self {
            path_mod(ident)            |
            path_name(ident)           |
            path_pretty_name(ident, _) => ident
        }
    }
}

pub type path = ~[path_elt];

pub fn path_to_str_with_sep(p: &[path_elt], sep: &str, itr: @ident_interner)
                         -> ~str {
    let strs = p.map(|e| {
        match *e {
            path_mod(s) | path_name(s) | path_pretty_name(s, _) => {
                itr.get(s.name)
            }
        }
    });
    strs.connect(sep)
}

pub fn path_ident_to_str(p: &path, i: Ident, itr: @ident_interner) -> ~str {
    if p.is_empty() {
        itr.get(i.name).to_owned()
    } else {
        format!("{}::{}", path_to_str(*p, itr), itr.get(i.name))
    }
}

pub fn path_to_str(p: &[path_elt], itr: @ident_interner) -> ~str {
    path_to_str_with_sep(p, "::", itr)
}

pub fn path_elt_to_str(pe: path_elt, itr: @ident_interner) -> ~str {
    match pe {
        path_mod(s) | path_name(s) | path_pretty_name(s, _) => {
            itr.get(s.name).to_owned()
        }
    }
}

/// write a "pretty" version of `ty` to `out`. This is designed so
/// that symbols of `impl`'d methods give some hint of where they came
/// from, even if it's hard to read (previously they would all just be
/// listed as `__extensions__::method_name::hash`, with no indication
/// of the type).
// XXX: these dollar signs and the names in general are actually a
//      relic of $ being one of the very few valid symbol names on
//      unix. These kinds of details shouldn't be exposed way up here
//      in the ast.
fn pretty_ty(ty: &Ty, itr: @ident_interner, out: &mut ~str) {
    let (prefix, subty) = match ty.node {
        ty_uniq(ty) => ("$UP$", &*ty),
        ty_box(ty) => ("$SP$", &*ty),
        ty_ptr(mt { ty, mutbl }) => (if mutbl == MutMutable {"$RPmut$"} else {"$RP$"},
                                     &*ty),
        ty_rptr(_, mt { ty, mutbl }) => (if mutbl == MutMutable {"$BPmut$"} else {"$BP$"},
                                      &*ty),

        ty_vec(ty) => ("$VEC$", &*ty),
        ty_fixed_length_vec(ty, _) => ("$FIXEDVEC$", &*ty),

        // these can't be represented as <prefix><contained ty>, so
        // need custom handling.
        ty_nil => { out.push_str("$NIL$"); return }
        ty_path(ref path, _, _) => {
                        out.push_str(itr.get(path.segments.last().identifier.name));
                        return
                    }
        ty_tup(ref tys) => {
            out.push_str(format!("$TUP_{}$", tys.len()));
            for subty in tys.iter() {
                pretty_ty(*subty, itr, out);
                out.push_char('$');
            }
            return
        }

        // meh, better than nothing.
        ty_bot => { out.push_str("$BOT$"); return }
        ty_closure(..) => { out.push_str("$CLOSURE$"); return }
        ty_bare_fn(..) => { out.push_str("$FN$"); return }
        ty_typeof(..) => { out.push_str("$TYPEOF$"); return }
        ty_infer(..) => { out.push_str("$INFER$"); return }

    };

    out.push_str(prefix);
    pretty_ty(subty, itr, out);
}

pub fn impl_pretty_name(trait_ref: &Option<trait_ref>, ty: &Ty) -> path_elt {
    let itr = get_ident_interner();

    let hash = (trait_ref, ty).hash();
    let mut pretty;
    match *trait_ref {
        None => pretty = ~"",
        Some(ref trait_ref) => {
            pretty = itr.get(trait_ref.path.segments.last().identifier.name).to_owned();
            pretty.push_char('$');
        }
    };
    pretty_ty(ty, itr, &mut pretty);

    path_pretty_name(Ident::new(itr.gensym(pretty)), hash)
}

#[deriving(Clone)]
pub enum ast_node {
    node_item(@item, @path),
    node_foreign_item(@foreign_item, AbiSet, visibility, @path),
    node_trait_method(@trait_method, DefId /* trait did */,
                      @path /* path to the trait */),
    node_method(@method, DefId /* impl did */, @path /* path to the impl */),

    /// node_variant represents a variant of an enum, e.g., for
    /// `enum A { B, C, D }`, there would be a node_item for `A`, and a
    /// node_variant item for each of `B`, `C`, and `D`.
    node_variant(P<variant>, @item, @path),
    node_expr(@Expr),
    node_stmt(@Stmt),
    node_arg(@Pat),
    // HACK(eddyb) should always be a pattern, but `self` is not, and thus it
    // is identified only by an ident and no span is available. In all other
    // cases, node_span will return the proper span (required by borrowck).
    node_local(Ident, Option<@Pat>),
    node_block(P<Block>),

    /// node_struct_ctor represents a tuple struct.
    node_struct_ctor(@struct_def, @item, @path),
    node_callee_scope(@Expr)
}

impl ast_node {
    pub fn with_attrs<T>(&self, f: |Option<&[Attribute]>| -> T) -> T {
        let attrs = match *self {
            node_item(i, _) => Some(i.attrs.as_slice()),
            node_foreign_item(fi, _, _, _) => Some(fi.attrs.as_slice()),
            node_trait_method(tm, _, _) => match *tm {
                required(ref type_m) => Some(type_m.attrs.as_slice()),
                provided(m) => Some(m.attrs.as_slice())
            },
            node_method(m, _, _) => Some(m.attrs.as_slice()),
            node_variant(ref v, _, _) => Some(v.node.attrs.as_slice()),
            // unit/tuple structs take the attributes straight from
            // the struct definition.
            node_struct_ctor(_, strct, _) => Some(strct.attrs.as_slice()),
            _ => None
        };
        f(attrs)
    }
}

pub type map = @RefCell<HashMap<NodeId, ast_node>>;

pub trait FoldOps {
    fn new_id(&self, id: ast::NodeId) -> ast::NodeId {
        id
    }
    fn new_span(&self, span: Span) -> Span {
        span
    }
}

pub struct Ctx<F> {
    map: map,
    path: path,
    diag: @SpanHandler,
    fold_ops: F
}

impl<F> Ctx<F> {
    fn insert(&self, id: ast::NodeId, node: ast_node) {
        let mut map = self.map.borrow_mut();
        map.get().insert(id, node);
    }

    fn map_self(&self, m: @method) {
        self.insert(m.self_id, node_local(special_idents::self_, None));
    }
}

impl<F: FoldOps> ast_fold for Ctx<F> {
    fn new_id(&mut self, id: ast::NodeId) -> ast::NodeId {
        self.fold_ops.new_id(id)
    }

    fn new_span(&mut self, span: Span) -> Span {
        self.fold_ops.new_span(span)
    }

    fn fold_item(&mut self, i: @item) -> SmallVector<@item> {
        // clone is FIXME #2543
        let item_path = @self.path.clone();
        self.path.push(match i.node {
            item_impl(_, ref maybe_trait, ty, _) => {
                // Right now the ident on impls is __extensions__ which isn't
                // very pretty when debugging, so attempt to select a better
                // name to use.
                impl_pretty_name(maybe_trait, ty)
            }
            item_mod(_) | item_foreign_mod(_) => path_mod(i.ident),
            _ => path_name(i.ident)
        });

        let i = fold::noop_fold_item(i, self).expect_one("expected one item");
        self.insert(i.id, node_item(i, item_path));

        match i.node {
            item_impl(_, _, _, ref ms) => {
                // clone is FIXME #2543
                let p = @self.path.clone();
                let impl_did = ast_util::local_def(i.id);
                for &m in ms.iter() {
                    self.insert(m.id, node_method(m, impl_did, p));
                    self.map_self(m);
                }

            }
            item_enum(ref enum_definition, _) => {
                // clone is FIXME #2543
                let p = @self.path.clone();
                for &v in enum_definition.variants.iter() {
                    self.insert(v.node.id, node_variant(v, i, p));
                }
            }
            item_foreign_mod(ref nm) => {
                for nitem in nm.items.iter() {
                    // Compute the visibility for this native item.
                    let visibility = match nitem.vis {
                        public => public,
                        private => private,
                        inherited => i.vis
                    };

                    self.insert(nitem.id,
                                // Anonymous extern mods go in the parent scope.
                                node_foreign_item(*nitem, nm.abis, visibility, item_path));
                }
            }
            item_struct(struct_def, _) => {
                // If this is a tuple-like struct, register the constructor.
                match struct_def.ctor_id {
                    None => {}
                    Some(ctor_id) => {
                        // clone is FIXME #2543
                        let p = @self.path.clone();
                        self.insert(ctor_id, node_struct_ctor(struct_def, i, p));
                    }
                }
            }
            item_trait(_, ref traits, ref methods) => {
                for t in traits.iter() {
                    self.insert(t.ref_id, node_item(i, item_path));
                }

                // clone is FIXME #2543
                let p = @self.path.clone();
                for tm in methods.iter() {
                    let d_id = ast_util::local_def(i.id);
                    match *tm {
                        required(ref m) => {
                            self.insert(m.id, node_trait_method(@(*tm).clone(), d_id, p));
                        }
                        provided(m) => {
                            self.insert(m.id, node_trait_method(@provided(m), d_id, p));
                            self.map_self(m);
                        }
                    }
                }
            }
            _ => {}
        }

        self.path.pop();

        SmallVector::one(i)
    }

    fn fold_pat(&mut self, pat: @Pat) -> @Pat {
        let pat = fold::noop_fold_pat(pat, self);
        match pat.node {
            PatIdent(_, ref path, _) => {
                // Note: this is at least *potentially* a pattern...
                self.insert(pat.id, node_local(ast_util::path_to_ident(path), Some(pat)));
            }
            _ => {}
        }

        pat
    }

    fn fold_expr(&mut self, expr: @Expr) -> @Expr {
        let expr = fold::noop_fold_expr(expr, self);

        self.insert(expr.id, node_expr(expr));

        // Expressions which are or might be calls:
        {
            let r = expr.get_callee_id();
            for callee_id in r.iter() {
                self.insert(*callee_id, node_callee_scope(expr));
            }
        }

        expr
    }

    fn fold_stmt(&mut self, stmt: &Stmt) -> SmallVector<@Stmt> {
        let stmt = fold::noop_fold_stmt(stmt, self).expect_one("expected one statement");
        self.insert(ast_util::stmt_id(stmt), node_stmt(stmt));
        SmallVector::one(stmt)
    }

    fn fold_method(&mut self, m: @method) -> @method {
        self.path.push(path_name(m.ident));
        let m = fold::noop_fold_method(m, self);
        self.path.pop();
        m
    }

    fn fold_fn_decl(&mut self, decl: &fn_decl) -> P<fn_decl> {
        let decl = fold::noop_fold_fn_decl(decl, self);
        for a in decl.inputs.iter() {
            self.insert(a.id, node_arg(a.pat));
        }
        decl
    }

    fn fold_block(&mut self, block: P<Block>) -> P<Block> {
        let block = fold::noop_fold_block(block, self);
        self.insert(block.id, node_block(block));
        block
    }
}

pub fn map_crate<F: 'static + FoldOps>(diag: @SpanHandler, c: Crate,
                                       fold_ops: F) -> (Crate, map) {
    let mut cx = Ctx {
        map: @RefCell::new(HashMap::new()),
        path: ~[],
        diag: diag,
        fold_ops: fold_ops
    };
    (cx.fold_crate(c), cx.map)
}

// Used for items loaded from external crate that are being inlined into this
// crate.  The `path` should be the path to the item but should not include
// the item itself.
pub fn map_decoded_item<F: 'static + FoldOps>(diag: @SpanHandler,
                                              map: map,
                                              path: path,
                                              fold_ops: F,
                                              fold_ii: |&mut Ctx<F>| -> inlined_item)
                                              -> inlined_item {
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
        ii_item(..) => {} // fallthrough
        ii_foreign(i) => {
            cx.insert(i.id, node_foreign_item(i,
                                              AbiSet::Intrinsic(),
                                              i.vis,    // Wrong but OK
                                              @path));
        }
        ii_method(impl_did, is_provided, m) => {
            let entry = if is_provided {
                node_trait_method(@provided(m), impl_did, @path)
            } else {
                node_method(m, impl_did, @path)
            };
            cx.insert(m.id, entry);
            cx.map_self(m);
        }
    }

    ii
}

pub fn node_id_to_str(map: map, id: NodeId, itr: @ident_interner) -> ~str {
    let map = map.borrow();
    match map.get().find(&id) {
      None => {
        format!("unknown node (id={})", id)
      }
      Some(&node_item(item, path)) => {
        let path_str = path_ident_to_str(path, item.ident, itr);
        let item_str = match item.node {
          item_static(..) => ~"static",
          item_fn(..) => ~"fn",
          item_mod(..) => ~"mod",
          item_foreign_mod(..) => ~"foreign mod",
          item_ty(..) => ~"ty",
          item_enum(..) => ~"enum",
          item_struct(..) => ~"struct",
          item_trait(..) => ~"trait",
          item_impl(..) => ~"impl",
          item_mac(..) => ~"macro"
        };
        format!("{} {} (id={})", item_str, path_str, id)
      }
      Some(&node_foreign_item(item, abi, _, path)) => {
        format!("foreign item {} with abi {:?} (id={})",
             path_ident_to_str(path, item.ident, itr), abi, id)
      }
      Some(&node_method(m, _, path)) => {
        format!("method {} in {} (id={})",
             itr.get(m.ident.name), path_to_str(*path, itr), id)
      }
      Some(&node_trait_method(ref tm, _, path)) => {
        let m = ast_util::trait_method_to_ty_method(&**tm);
        format!("method {} in {} (id={})",
             itr.get(m.ident.name), path_to_str(*path, itr), id)
      }
      Some(&node_variant(ref variant, _, path)) => {
        format!("variant {} in {} (id={})",
             itr.get(variant.node.name.name), path_to_str(*path, itr), id)
      }
      Some(&node_expr(expr)) => {
        format!("expr {} (id={})", pprust::expr_to_str(expr, itr), id)
      }
      Some(&node_callee_scope(expr)) => {
        format!("callee_scope {} (id={})", pprust::expr_to_str(expr, itr), id)
      }
      Some(&node_stmt(stmt)) => {
        format!("stmt {} (id={})",
             pprust::stmt_to_str(stmt, itr), id)
      }
      Some(&node_arg(pat)) => {
        format!("arg {} (id={})", pprust::pat_to_str(pat, itr), id)
      }
      Some(&node_local(ident, _)) => {
        format!("local (id={}, name={})", id, itr.get(ident.name))
      }
      Some(&node_block(block)) => {
        format!("block {} (id={})", pprust::block_to_str(block, itr), id)
      }
      Some(&node_struct_ctor(_, _, path)) => {
        format!("struct_ctor {} (id={})", path_to_str(*path, itr), id)
      }
    }
}

pub fn node_item_query<Result>(items: map, id: NodeId, query: |@item| -> Result, error_msg: ~str)
                       -> Result {
    let items = items.borrow();
    match items.get().find(&id) {
        Some(&node_item(it, _)) => query(it),
        _ => fail!("{}", error_msg)
    }
}

pub fn node_span(items: map,
                 id: ast::NodeId)
                 -> Span {
    let items = items.borrow();
    match items.get().find(&id) {
        Some(&node_item(item, _)) => item.span,
        Some(&node_foreign_item(foreign_item, _, _, _)) => foreign_item.span,
        Some(&node_trait_method(@required(ref type_method), _, _)) => type_method.span,
        Some(&node_trait_method(@provided(ref method), _, _)) => method.span,
        Some(&node_method(method, _, _)) => method.span,
        Some(&node_variant(variant, _, _)) => variant.span,
        Some(&node_expr(expr)) => expr.span,
        Some(&node_stmt(stmt)) => stmt.span,
        Some(&node_arg(pat)) => pat.span,
        Some(&node_local(_, pat)) => match pat {
            Some(pat) => pat.span,
            None => fail!("node_span: cannot get span from node_local (likely `self`)")
        },
        Some(&node_block(block)) => block.span,
        Some(&node_struct_ctor(_, item, _)) => item.span,
        Some(&node_callee_scope(expr)) => expr.span,
        None => fail!("node_span: could not find id {}", id),
    }
}
