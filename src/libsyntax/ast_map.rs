// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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
use ast_util::{inlined_item_utils, stmt_id};
use ast_util;
use codemap::Span;
use codemap;
use diagnostic::span_handler;
use parse::token::get_ident_interner;
use parse::token::ident_interner;
use parse::token::special_idents;
use print::pprust;
use visit::{Visitor, fn_kind};
use visit;

use std::hashmap::HashMap;
use std::vec;

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

pub type path = ~[path_elt];

pub fn path_to_str_with_sep(p: &[path_elt], sep: &str, itr: @ident_interner)
                         -> ~str {
    let strs = do p.map |e| {
        match *e {
            path_mod(s) | path_name(s) | path_pretty_name(s, _) => {
                itr.get(s.name)
            }
        }
    };
    strs.connect(sep)
}

pub fn path_ident_to_str(p: &path, i: Ident, itr: @ident_interner) -> ~str {
    if p.is_empty() {
        itr.get(i.name).to_owned()
    } else {
        fmt!("%s::%s", path_to_str(*p, itr), itr.get(i.name))
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

pub fn impl_pretty_name(trait_ref: &Option<trait_ref>,
                        ty: &Ty, default: Ident) -> path_elt {
    let itr = get_ident_interner();
    let ty_ident = match ty.node {
        ty_path(ref path, _, _) => path.segments.last().identifier,
        _ => default
    };
    let hash = (trait_ref, ty).hash();
    match *trait_ref {
        None => path_pretty_name(ty_ident, hash),
        Some(ref trait_ref) => {
            // XXX: this dollar sign is actually a relic of being one of the
            //      very few valid symbol names on unix. These kinds of
            //      details shouldn't be exposed way up here in the ast.
            let s = fmt!("%s$%s",
                         itr.get(trait_ref.path.segments.last().identifier.name),
                         itr.get(ty_ident.name));
            path_pretty_name(Ident::new(itr.gensym(s)), hash)
        }
    }
}

#[deriving(Clone)]
pub enum ast_node {
    node_item(@item, @path),
    node_foreign_item(@foreign_item, AbiSet, visibility, @path),
    node_trait_method(@trait_method, DefId /* trait did */,
                      @path /* path to the trait */),
    node_method(@method, DefId /* impl did */, @path /* path to the impl */),
    node_variant(variant, @item, @path),
    node_expr(@Expr),
    node_stmt(@Stmt),
    node_arg(@Pat),
    node_local(Ident),
    node_block(Block),
    node_struct_ctor(@struct_def, @item, @path),
    node_callee_scope(@Expr)
}

impl ast_node {
    pub fn with_attrs<T>(&self, f: &fn(Option<&[Attribute]>) -> T) -> T {
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

pub type map = @mut HashMap<NodeId, ast_node>;

pub struct Ctx {
    map: map,
    path: path,
    diag: @mut span_handler,
}

impl Ctx {
    fn extend(&self, elt: path_elt) -> @path {
        @vec::append(self.path.clone(), [elt])
    }

    fn map_method(&mut self,
                  impl_did: DefId,
                  impl_path: @path,
                  m: @method,
                  is_provided: bool) {
        let entry = if is_provided {
            node_trait_method(@provided(m), impl_did, impl_path)
        } else {
            node_method(m, impl_did, impl_path)
        };
        self.map.insert(m.id, entry);
        self.map.insert(m.self_id, node_local(special_idents::self_));
    }

    fn map_struct_def(&mut self,
                      struct_def: @ast::struct_def,
                      parent_node: ast_node,
                      ident: ast::Ident) {
        let p = self.extend(path_name(ident));

        // If this is a tuple-like struct, register the constructor.
        match struct_def.ctor_id {
            None => {}
            Some(ctor_id) => {
                match parent_node {
                    node_item(item, _) => {
                        self.map.insert(ctor_id,
                                        node_struct_ctor(struct_def,
                                                         item,
                                                         p));
                    }
                    _ => fail!("struct def parent wasn't an item")
                }
            }
        }
    }

    fn map_expr(&mut self, ex: @Expr) {
        self.map.insert(ex.id, node_expr(ex));

        // Expressions which are or might be calls:
        {
            let r = ex.get_callee_id();
            for callee_id in r.iter() {
                self.map.insert(*callee_id, node_callee_scope(ex));
            }
        }

        visit::walk_expr(self, ex, ());
    }

    fn map_fn(&mut self,
              fk: &visit::fn_kind,
              decl: &fn_decl,
              body: &Block,
              sp: codemap::Span,
              id: NodeId) {
        for a in decl.inputs.iter() {
            self.map.insert(a.id, node_arg(a.pat));
        }
        match *fk {
            visit::fk_method(name, _, _) => { self.path.push(path_name(name)) }
            _ => {}
        }
        visit::walk_fn(self, fk, decl, body, sp, id, ());
        match *fk {
            visit::fk_method(*) => { self.path.pop(); }
            _ => {}
        }
    }

    fn map_stmt(&mut self, stmt: @Stmt) {
        self.map.insert(stmt_id(stmt), node_stmt(stmt));
        visit::walk_stmt(self, stmt, ());
    }

    fn map_block(&mut self, b: &Block) {
        // clone is FIXME #2543
        self.map.insert(b.id, node_block((*b).clone()));
        visit::walk_block(self, b, ());
    }

    fn map_pat(&mut self, pat: @Pat) {
        match pat.node {
            PatIdent(_, ref path, _) => {
                // Note: this is at least *potentially* a pattern...
                self.map.insert(pat.id,
                                node_local(ast_util::path_to_ident(path)));
            }
            _ => ()
        }

        visit::walk_pat(self, pat, ());
    }
}

impl Visitor<()> for Ctx {
    fn visit_item(&mut self, i: @item, _: ()) {
        // clone is FIXME #2543
        let item_path = @self.path.clone();
        self.map.insert(i.id, node_item(i, item_path));
        match i.node {
            item_impl(_, ref maybe_trait, ref ty, ref ms) => {
                // Right now the ident on impls is __extensions__ which isn't
                // very pretty when debugging, so attempt to select a better
                // name to use.
                let elt = impl_pretty_name(maybe_trait, ty, i.ident);

                let impl_did = ast_util::local_def(i.id);
                for m in ms.iter() {
                    let extended = { self.extend(elt) };
                    self.map_method(impl_did, extended, *m, false)
                }

                self.path.push(elt);
            }
            item_enum(ref enum_definition, _) => {
                for v in (*enum_definition).variants.iter() {
                    let elt = path_name(i.ident);
                    // FIXME #2543: bad clone
                    self.map.insert(v.node.id,
                                    node_variant((*v).clone(),
                                                 i,
                                                 self.extend(elt)));
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

                    self.map.insert(nitem.id,
                                    node_foreign_item(*nitem,
                                                      nm.abis,
                                                      visibility,
                                                      // FIXME (#2543)
                                                      if nm.sort ==
                                                            ast::named {
                                                          let e = path_name(
                                                              i.ident);
                                                          self.extend(e)
                                                      } else {
                                                        // Anonymous extern
                                                        // mods go in the
                                                        // parent scope.
                                                        @self.path.clone()
                                                      }));
                }
            }
            item_struct(struct_def, _) => {
                self.map_struct_def(struct_def,
                                    node_item(i, item_path),
                                    i.ident)
            }
            item_trait(_, ref traits, ref methods) => {
                for p in traits.iter() {
                    self.map.insert(p.ref_id, node_item(i, item_path));
                }
                for tm in methods.iter() {
                    let ext = { self.extend(path_name(i.ident)) };
                    let d_id = ast_util::local_def(i.id);
                    match *tm {
                        required(ref m) => {
                            let entry =
                                node_trait_method(@(*tm).clone(), d_id, ext);
                            self.map.insert(m.id, entry);
                        }
                        provided(m) => {
                            self.map_method(d_id, ext, m, true);
                        }
                    }
                }
            }
            _ => {}
        }

        match i.node {
            item_mod(_) | item_foreign_mod(_) => {
                self.path.push(path_mod(i.ident));
            }
            item_impl(*) => {} // this was guessed above.
            _ => self.path.push(path_name(i.ident))
        }
        visit::walk_item(self, i, ());
        self.path.pop();
    }

    fn visit_pat(&mut self, pat: @Pat, _: ()) {
        self.map_pat(pat);
        visit::walk_pat(self, pat, ())
    }

    fn visit_expr(&mut self, expr: @Expr, _: ()) {
        self.map_expr(expr)
    }

    fn visit_stmt(&mut self, stmt: @Stmt, _: ()) {
        self.map_stmt(stmt)
    }

    fn visit_fn(&mut self,
                function_kind: &fn_kind,
                function_declaration: &fn_decl,
                block: &Block,
                span: Span,
                node_id: NodeId,
                _: ()) {
        self.map_fn(function_kind, function_declaration, block, span, node_id)
    }

    fn visit_block(&mut self, block: &Block, _: ()) {
        self.map_block(block)
    }

    // XXX: Methods below can become default methods.

    fn visit_mod(&mut self, module: &_mod, _: Span, _: NodeId, _: ()) {
        visit::walk_mod(self, module, ())
    }

    fn visit_view_item(&mut self, view_item: &view_item, _: ()) {
        visit::walk_view_item(self, view_item, ())
    }

    fn visit_foreign_item(&mut self, foreign_item: @foreign_item, _: ()) {
        visit::walk_foreign_item(self, foreign_item, ())
    }

    fn visit_local(&mut self, local: @Local, _: ()) {
        visit::walk_local(self, local, ())
    }

    fn visit_arm(&mut self, arm: &Arm, _: ()) {
        visit::walk_arm(self, arm, ())
    }

    fn visit_decl(&mut self, decl: @Decl, _: ()) {
        visit::walk_decl(self, decl, ())
    }

    fn visit_expr_post(&mut self, _: @Expr, _: ()) {
        // Empty!
    }

    fn visit_ty(&mut self, typ: &Ty, _: ()) {
        visit::walk_ty(self, typ, ())
    }

    fn visit_generics(&mut self, generics: &Generics, _: ()) {
        visit::walk_generics(self, generics, ())
    }

    fn visit_fn(&mut self,
                function_kind: &fn_kind,
                function_declaration: &fn_decl,
                block: &Block,
                span: Span,
                node_id: NodeId,
                _: ()) {
        visit::walk_fn(self,
                        function_kind,
                        function_declaration,
                        block,
                        span,
                        node_id,
                        ())
    }

    fn visit_ty_method(&mut self, ty_method: &TypeMethod, _: ()) {
        visit::walk_ty_method(self, ty_method, ())
    }

    fn visit_trait_method(&mut self, trait_method: &trait_method, _: ()) {
        visit::walk_trait_method(self, trait_method, ())
    }

    fn visit_struct_def(&mut self,
                        struct_def: @struct_def,
                        ident: Ident,
                        generics: &Generics,
                        node_id: NodeId,
                        _: ()) {
        visit::walk_struct_def(self,
                                struct_def,
                                ident,
                                generics,
                                node_id,
                                ())
    }

    fn visit_struct_field(&mut self, struct_field: @struct_field, _: ()) {
        visit::walk_struct_field(self, struct_field, ())
    }
}

pub fn map_crate(diag: @mut span_handler, c: &Crate) -> map {
    let cx = @mut Ctx {
        map: @mut HashMap::new(),
        path: ~[],
        diag: diag,
    };
    visit::walk_crate(cx, c, ());
    cx.map
}

// Used for items loaded from external crate that are being inlined into this
// crate.  The `path` should be the path to the item but should not include
// the item itself.
pub fn map_decoded_item(diag: @mut span_handler,
                        map: map,
                        path: path,
                        ii: &inlined_item) {
    // I believe it is ok for the local IDs of inlined items from other crates
    // to overlap with the local ids from this crate, so just generate the ids
    // starting from 0.
    let cx = @mut Ctx {
        map: map,
        path: path.clone(),
        diag: diag,
    };

    // methods get added to the AST map when their impl is visited.  Since we
    // don't decode and instantiate the impl, but just the method, we have to
    // add it to the table now:
    match *ii {
        ii_item(*) => {} // fallthrough
        ii_foreign(i) => {
            cx.map.insert(i.id, node_foreign_item(i,
                                                  AbiSet::Intrinsic(),
                                                  i.vis,    // Wrong but OK
                                                  @path));
        }
        ii_method(impl_did, is_provided, m) => {
            cx.map_method(impl_did, @path, m, is_provided);
        }
    }

    // visit the item / method contents and add those to the map:
    ii.accept((), cx);
}

pub fn node_id_to_str(map: map, id: NodeId, itr: @ident_interner) -> ~str {
    match map.find(&id) {
      None => {
        fmt!("unknown node (id=%d)", id)
      }
      Some(&node_item(item, path)) => {
        let path_str = path_ident_to_str(path, item.ident, itr);
        let item_str = match item.node {
          item_static(*) => ~"static",
          item_fn(*) => ~"fn",
          item_mod(*) => ~"mod",
          item_foreign_mod(*) => ~"foreign mod",
          item_ty(*) => ~"ty",
          item_enum(*) => ~"enum",
          item_struct(*) => ~"struct",
          item_trait(*) => ~"trait",
          item_impl(*) => ~"impl",
          item_mac(*) => ~"macro"
        };
        fmt!("%s %s (id=%?)", item_str, path_str, id)
      }
      Some(&node_foreign_item(item, abi, _, path)) => {
        fmt!("foreign item %s with abi %? (id=%?)",
             path_ident_to_str(path, item.ident, itr), abi, id)
      }
      Some(&node_method(m, _, path)) => {
        fmt!("method %s in %s (id=%?)",
             itr.get(m.ident.name), path_to_str(*path, itr), id)
      }
      Some(&node_trait_method(ref tm, _, path)) => {
        let m = ast_util::trait_method_to_ty_method(&**tm);
        fmt!("method %s in %s (id=%?)",
             itr.get(m.ident.name), path_to_str(*path, itr), id)
      }
      Some(&node_variant(ref variant, _, path)) => {
        fmt!("variant %s in %s (id=%?)",
             itr.get(variant.node.name.name), path_to_str(*path, itr), id)
      }
      Some(&node_expr(expr)) => {
        fmt!("expr %s (id=%?)", pprust::expr_to_str(expr, itr), id)
      }
      Some(&node_callee_scope(expr)) => {
        fmt!("callee_scope %s (id=%?)", pprust::expr_to_str(expr, itr), id)
      }
      Some(&node_stmt(stmt)) => {
        fmt!("stmt %s (id=%?)",
             pprust::stmt_to_str(stmt, itr), id)
      }
      Some(&node_arg(pat)) => {
        fmt!("arg %s (id=%?)", pprust::pat_to_str(pat, itr), id)
      }
      Some(&node_local(ident)) => {
        fmt!("local (id=%?, name=%s)", id, itr.get(ident.name))
      }
      Some(&node_block(ref block)) => {
        fmt!("block %s (id=%?)", pprust::block_to_str(block, itr), id)
      }
      Some(&node_struct_ctor(_, _, path)) => {
        fmt!("struct_ctor %s (id=%?)", path_to_str(*path, itr), id)
      }
    }
}

pub fn node_item_query<Result>(items: map, id: NodeId,
                               query: &fn(@item) -> Result,
                               error_msg: ~str) -> Result {
    match items.find(&id) {
        Some(&node_item(it, _)) => query(it),
        _ => fail!(error_msg)
    }
}
