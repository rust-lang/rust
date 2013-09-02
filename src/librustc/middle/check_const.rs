// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use driver::session::Session;
use middle::resolve;
use middle::ty;
use middle::typeck;
use util::ppaux;

use syntax::ast::*;
use syntax::codemap;
use syntax::{ast_util, ast_map};
use syntax::visit::Visitor;
use syntax::visit;

struct CheckCrateVisitor {
    sess: Session,
    ast_map: ast_map::map,
    def_map: resolve::DefMap,
    method_map: typeck::method_map,
    tcx: ty::ctxt,
}

impl Visitor<bool> for CheckCrateVisitor {
    fn visit_item(&mut self, i:@item, env:bool) {
        check_item(self, self.sess, self.ast_map, self.def_map, i, env);
    }
    fn visit_pat(&mut self, p:@Pat, env:bool) {
        check_pat(self, p, env);
    }
    fn visit_expr(&mut self, ex:@Expr, env:bool) {
        check_expr(self, self.sess, self.def_map, self.method_map,
                   self.tcx, ex, env);
    }
}

pub fn check_crate(sess: Session,
                   crate: &Crate,
                   ast_map: ast_map::map,
                   def_map: resolve::DefMap,
                   method_map: typeck::method_map,
                   tcx: ty::ctxt) {
    let mut v = CheckCrateVisitor {
        sess: sess,
        ast_map: ast_map,
        def_map: def_map,
        method_map: method_map,
        tcx: tcx,
    };
    visit::walk_crate(&mut v, crate, false);
    sess.abort_if_errors();
}

pub fn check_item(v: &mut CheckCrateVisitor,
                  sess: Session,
                  ast_map: ast_map::map,
                  def_map: resolve::DefMap,
                  it: @item,
                  _is_const: bool) {
    match it.node {
      item_static(_, _, ex) => {
        v.visit_expr(ex, true);
        check_item_recursion(sess, ast_map, def_map, it);
      }
      item_enum(ref enum_definition, _) => {
        for var in (*enum_definition).variants.iter() {
            for ex in var.node.disr_expr.iter() {
                v.visit_expr(*ex, true);
            }
        }
      }
      _ => visit::walk_item(v, it, false)
    }
}

pub fn check_pat(v: &mut CheckCrateVisitor, p: @Pat, _is_const: bool) {
    fn is_str(e: @Expr) -> bool {
        match e.node {
            ExprVstore(
                @Expr { node: ExprLit(@codemap::Spanned {
                    node: lit_str(_),
                    _}),
                       _ },
                ExprVstoreUniq
            ) => true,
            _ => false
        }
    }
    match p.node {
      // Let through plain ~-string literals here
      PatLit(a) => if !is_str(a) { v.visit_expr(a, true); },
      PatRange(a, b) => {
        if !is_str(a) { v.visit_expr(a, true); }
        if !is_str(b) { v.visit_expr(b, true); }
      }
      _ => visit::walk_pat(v, p, false)
    }
}

pub fn check_expr(v: &mut CheckCrateVisitor,
                  sess: Session,
                  def_map: resolve::DefMap,
                  method_map: typeck::method_map,
                  tcx: ty::ctxt,
                  e: @Expr,
                  is_const: bool) {
    if is_const {
        match e.node {
          ExprUnary(_, UnDeref, _) => { }
          ExprUnary(_, UnBox(_), _) | ExprUnary(_, UnUniq, _) => {
            sess.span_err(e.span,
                          "disallowed operator in constant expression");
            return;
          }
          ExprLit(@codemap::Spanned {node: lit_str(_), _}) => { }
          ExprBinary(*) | ExprUnary(*) => {
            if method_map.contains_key(&e.id) {
                sess.span_err(e.span, "user-defined operators are not \
                                       allowed in constant expressions");
            }
          }
          ExprLit(_) => (),
          ExprCast(_, _) => {
            let ety = ty::expr_ty(tcx, e);
            if !ty::type_is_numeric(ety) && !ty::type_is_unsafe_ptr(ety) {
                sess.span_err(e.span, ~"can not cast to `" +
                              ppaux::ty_to_str(tcx, ety) +
                              "` in a constant expression");
            }
          }
          ExprPath(ref pth) => {
            // NB: In the future you might wish to relax this slightly
            // to handle on-demand instantiation of functions via
            // foo::<bar> in a const. Currently that is only done on
            // a path in trans::callee that only works in block contexts.
            if !pth.segments.iter().all(|segment| segment.types.is_empty()) {
                sess.span_err(
                    e.span, "paths in constants may only refer to \
                             items without type parameters");
            }
            match def_map.find(&e.id) {
              Some(&DefStatic(*)) |
              Some(&DefFn(_, _)) |
              Some(&DefVariant(_, _)) |
              Some(&DefStruct(_)) => { }

              Some(&def) => {
                debug!("(checking const) found bad def: %?", def);
                sess.span_err(
                    e.span,
                    "paths in constants may only refer to \
                     constants or functions");
              }
              None => {
                sess.span_bug(e.span, "unbound path in const?!");
              }
            }
          }
          ExprCall(callee, _, NoSugar) => {
            match def_map.find(&callee.id) {
                Some(&DefStruct(*)) => {}    // OK.
                Some(&DefVariant(*)) => {}    // OK.
                _ => {
                    sess.span_err(
                        e.span,
                        "function calls in constants are limited to \
                         struct and enum constructors");
                }
            }
          }
          ExprParen(e) => { check_expr(v, sess, def_map, method_map,
                                        tcx, e, is_const); }
          ExprVstore(_, ExprVstoreSlice) |
          ExprVec(_, MutImmutable) |
          ExprAddrOf(MutImmutable, _) |
          ExprField(*) |
          ExprIndex(*) |
          ExprTup(*) |
          ExprRepeat(*) |
          ExprStruct(*) => { }
          ExprAddrOf(*) => {
                sess.span_err(
                    e.span,
                    "borrowed pointers in constants may only refer to \
                     immutable values");
          }
          _ => {
            sess.span_err(e.span,
                          "constant contains unimplemented expression type");
            return;
          }
        }
    }
    match e.node {
      ExprLit(@codemap::Spanned {node: lit_int(v, t), _}) => {
        if t != ty_char {
            if (v as u64) > ast_util::int_ty_max(
                if t == ty_i { sess.targ_cfg.int_type } else { t }) {
                sess.span_err(e.span, "literal out of range for its type");
            }
        }
      }
      ExprLit(@codemap::Spanned {node: lit_uint(v, t), _}) => {
        if v > ast_util::uint_ty_max(
            if t == ty_u { sess.targ_cfg.uint_type } else { t }) {
            sess.span_err(e.span, "literal out of range for its type");
        }
      }
      _ => ()
    }
    visit::walk_expr(v, e, is_const);
}

#[deriving(Clone)]
struct env {
    root_it: @item,
    sess: Session,
    ast_map: ast_map::map,
    def_map: resolve::DefMap,
    idstack: @mut ~[NodeId]
}

struct CheckItemRecursionVisitor;

// Make sure a const item doesn't recursively refer to itself
// FIXME: Should use the dependency graph when it's available (#1356)
pub fn check_item_recursion(sess: Session,
                            ast_map: ast_map::map,
                            def_map: resolve::DefMap,
                            it: @item) {
    let env = env {
        root_it: it,
        sess: sess,
        ast_map: ast_map,
        def_map: def_map,
        idstack: @mut ~[]
    };

    let mut visitor = CheckItemRecursionVisitor;
    visitor.visit_item(it, env);
}

impl Visitor<env> for CheckItemRecursionVisitor {
    fn visit_item(&mut self, it: @item, env: env) {
        if env.idstack.iter().any(|x| x == &(it.id)) {
            env.sess.span_fatal(env.root_it.span, "recursive constant");
        }
        env.idstack.push(it.id);
        visit::walk_item(self, it, env);
        env.idstack.pop();
    }

    fn visit_expr(&mut self, e: @Expr, env: env) {
        match e.node {
            ExprPath(*) => match env.def_map.find(&e.id) {
                Some(&DefStatic(def_id, _)) if ast_util::is_local(def_id) =>
                    match env.ast_map.get_copy(&def_id.node) {
                        ast_map::node_item(it, _) => {
                            self.visit_item(it, env);
                        }
                        _ => fail!("const not bound to an item")
                    },
                _ => ()
            },
            _ => ()
        }
        visit::walk_expr(self, e, env);
    }
}
