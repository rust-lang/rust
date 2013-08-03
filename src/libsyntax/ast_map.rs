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
use codemap::span;
use codemap;
use diagnostic::span_handler;
use parse::token::ident_interner;
use parse::token::special_idents;
use print::pprust;
use visit::{Visitor, fn_kind};
use visit;

use std::hashmap::HashMap;
use std::vec;

#[deriving(Clone, Eq)]
pub enum path_elt {
    path_mod(ident),
    path_name(ident)
}

pub type path = ~[path_elt];

pub fn path_to_str_with_sep(p: &[path_elt], sep: &str, itr: @ident_interner)
                         -> ~str {
    let strs = do p.map |e| {
        match *e {
          path_mod(s) => itr.get(s.name),
          path_name(s) => itr.get(s.name)
        }
    };
    strs.connect(sep)
}

pub fn path_ident_to_str(p: &path, i: ident, itr: @ident_interner) -> ~str {
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
        path_mod(s) => itr.get(s.name).to_owned(),
        path_name(s) => itr.get(s.name).to_owned()
    }
}

#[deriving(Clone)]
pub enum ast_node {
    node_item(@item, @path),
    node_foreign_item(@foreign_item, AbiSet, visibility, @path),
    node_trait_method(@trait_method, def_id /* trait did */,
                      @path /* path to the trait */),
    node_method(@method, def_id /* impl did */, @path /* path to the impl */),
    node_variant(variant, @item, @path),
    node_expr(@expr),
    node_stmt(@stmt),
    node_arg,
    node_local(ident),
    node_block(Block),
    node_struct_ctor(@struct_def, @item, @path),
    node_callee_scope(@expr)
}

pub type map = @mut HashMap<NodeId, ast_node>;

pub struct Ctx {
    map: map,
    path: path,
    diag: @span_handler,
}

impl Ctx {
    fn extend(@mut self, elt: ident) -> @path {
        @vec::append(self.path.clone(), [path_name(elt)])
    }

    fn map_method(@mut self,
                  impl_did: def_id,
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

    fn map_struct_def(@mut self,
                      struct_def: @ast::struct_def,
                      parent_node: ast_node,
                      ident: ast::ident) {
        let p = self.extend(ident);

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

    fn map_expr(@mut self, ex: @expr) {
        self.map.insert(ex.id, node_expr(ex));

        // Expressions which are or might be calls:
        {
            let r = ex.get_callee_id();
            foreach callee_id in r.iter() {
                self.map.insert(*callee_id, node_callee_scope(ex));
            }
        }

        visit::visit_expr(self as @Visitor<()>, ex, ());
    }

    fn map_fn(@mut self,
              fk: &visit::fn_kind,
              decl: &fn_decl,
              body: &Block,
              sp: codemap::span,
              id: NodeId) {
        foreach a in decl.inputs.iter() {
            self.map.insert(a.id, node_arg);
        }
        visit::visit_fn(self as @Visitor<()>, fk, decl, body, sp, id, ());
    }

    fn map_stmt(@mut self, stmt: @stmt) {
        self.map.insert(stmt_id(stmt), node_stmt(stmt));
        visit::visit_stmt(self as @Visitor<()>, stmt, ());
    }

    fn map_block(@mut self, b: &Block) {
        // clone is FIXME #2543
        self.map.insert(b.id, node_block((*b).clone()));
        visit::visit_block(self as @Visitor<()>, b, ());
    }

    fn map_pat(@mut self, pat: @pat) {
        match pat.node {
            pat_ident(_, ref path, _) => {
                // Note: this is at least *potentially* a pattern...
                self.map.insert(pat.id,
                                node_local(ast_util::path_to_ident(path)));
            }
            _ => ()
        }

        visit::visit_pat(self as @Visitor<()>, pat, ());
    }
}

impl Visitor<()> for Ctx {
    fn visit_item(@mut self, i: @item, _: ()) {
        // clone is FIXME #2543
        let item_path = @self.path.clone();
        self.map.insert(i.id, node_item(i, item_path));
        match i.node {
            item_impl(_, _, _, ref ms) => {
                let impl_did = ast_util::local_def(i.id);
                foreach m in ms.iter() {
                    self.map_method(impl_did, self.extend(i.ident), *m, false)
                }
            }
            item_enum(ref enum_definition, _) => {
                foreach v in (*enum_definition).variants.iter() {
                    // FIXME #2543: bad clone
                    self.map.insert(v.node.id,
                                    node_variant((*v).clone(),
                                                 i,
                                                 self.extend(i.ident)));
                }
            }
            item_foreign_mod(ref nm) => {
                foreach nitem in nm.items.iter() {
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
                                                        self.extend(i.ident)
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
                foreach p in traits.iter() {
                    self.map.insert(p.ref_id, node_item(i, item_path));
                }
                foreach tm in methods.iter() {
                    let id = ast_util::trait_method_to_ty_method(tm).id;
                    let d_id = ast_util::local_def(i.id);
                    self.map.insert(id,
                                    node_trait_method(@(*tm).clone(),
                                                      d_id,
                                                      item_path));
                }
            }
            _ => {}
        }

        match i.node {
            item_mod(_) | item_foreign_mod(_) => {
                self.path.push(path_mod(i.ident));
            }
            _ => self.path.push(path_name(i.ident))
        }
        visit::visit_item(self as @Visitor<()>, i, ());
        self.path.pop();
    }

    fn visit_pat(@mut self, pat: @pat, _: ()) {
        self.map_pat(pat);
        visit::visit_pat(self as @Visitor<()>, pat, ())
    }

    fn visit_expr(@mut self, expr: @expr, _: ()) {
        self.map_expr(expr)
    }

    fn visit_stmt(@mut self, stmt: @stmt, _: ()) {
        self.map_stmt(stmt)
    }

    fn visit_fn(@mut self,
                function_kind: &fn_kind,
                function_declaration: &fn_decl,
                block: &Block,
                span: span,
                node_id: NodeId,
                _: ()) {
        self.map_fn(function_kind, function_declaration, block, span, node_id)
    }

    fn visit_block(@mut self, block: &Block, _: ()) {
        self.map_block(block)
    }

    // XXX: Methods below can become default methods.

    fn visit_mod(@mut self, module: &_mod, _: span, _: NodeId, _: ()) {
        visit::visit_mod(self as @Visitor<()>, module, ())
    }

    fn visit_view_item(@mut self, view_item: &view_item, _: ()) {
        visit::visit_view_item(self as @Visitor<()>, view_item, ())
    }

    fn visit_foreign_item(@mut self, foreign_item: @foreign_item, _: ()) {
        visit::visit_foreign_item(self as @Visitor<()>, foreign_item, ())
    }

    fn visit_local(@mut self, local: @Local, _: ()) {
        visit::visit_local(self as @Visitor<()>, local, ())
    }

    fn visit_arm(@mut self, arm: &arm, _: ()) {
        visit::visit_arm(self as @Visitor<()>, arm, ())
    }

    fn visit_decl(@mut self, decl: @decl, _: ()) {
        visit::visit_decl(self as @Visitor<()>, decl, ())
    }

    fn visit_expr_post(@mut self, _: @expr, _: ()) {
        // Empty!
    }

    fn visit_ty(@mut self, typ: &Ty, _: ()) {
        visit::visit_ty(self as @Visitor<()>, typ, ())
    }

    fn visit_generics(@mut self, generics: &Generics, _: ()) {
        visit::visit_generics(self as @Visitor<()>, generics, ())
    }

    fn visit_fn(@mut self,
                function_kind: &fn_kind,
                function_declaration: &fn_decl,
                block: &Block,
                span: span,
                node_id: NodeId,
                _: ()) {
        visit::visit_fn(self as @Visitor<()>,
                        function_kind,
                        function_declaration,
                        block,
                        span,
                        node_id,
                        ())
    }

    fn visit_ty_method(@mut self, ty_method: &TypeMethod, _: ()) {
        visit::visit_ty_method(self as @Visitor<()>, ty_method, ())
    }

    fn visit_trait_method(@mut self, trait_method: &trait_method, _: ()) {
        visit::visit_trait_method(self as @Visitor<()>, trait_method, ())
    }

    fn visit_struct_def(@mut self,
                        struct_def: @struct_def,
                        ident: ident,
                        generics: &Generics,
                        node_id: NodeId,
                        _: ()) {
        visit::visit_struct_def(self as @Visitor<()>,
                                struct_def,
                                ident,
                                generics,
                                node_id,
                                ())
    }

    fn visit_struct_field(@mut self, struct_field: @struct_field, _: ()) {
        visit::visit_struct_field(self as @Visitor<()>, struct_field, ())
    }
}

pub fn map_crate(diag: @span_handler, c: &Crate) -> map {
    let cx = @mut Ctx {
        map: @mut HashMap::new(),
        path: ~[],
        diag: diag,
    };
    visit::visit_crate(cx as @Visitor<()>, c, ());
    cx.map
}

// Used for items loaded from external crate that are being inlined into this
// crate.  The `path` should be the path to the item but should not include
// the item itself.
pub fn map_decoded_item(diag: @span_handler,
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
    ii.accept((), cx as @Visitor<()>);
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
      Some(&node_arg) => {
        fmt!("arg (id=%?)", id)
      }
      Some(&node_local(ident)) => {
        fmt!("local (id=%?, name=%s)", id, itr.get(ident.name))
      }
      Some(&node_block(_)) => {
        fmt!("block")
      }
      Some(&node_struct_ctor(*)) => {
        fmt!("struct_ctor")
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
