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
use syntax::{ast_util, ast_map};
use syntax::visit::Visitor;
use syntax::visit;

pub struct CheckCrateVisitor<'a> {
    def_map: resolve::DefMap,
    method_map: typeck::MethodMap,
    tcx: &'a ty::ctxt,
}

impl<'a> Visitor<bool> for CheckCrateVisitor<'a> {
    fn visit_item(&mut self, i: &Item, env: bool) {
        check_item(self, i, env);
    }
    fn visit_pat(&mut self, p: &Pat, env: bool) {
        check_pat(self, p, env);
    }
    fn visit_expr(&mut self, ex: &Expr, env: bool) {
        check_expr(self, ex, env);
    }
}

pub fn check_crate(krate: &Crate,
                   def_map: resolve::DefMap,
                   method_map: typeck::MethodMap,
                   tcx: &ty::ctxt) {
    let mut v = CheckCrateVisitor {
        def_map: def_map,
        method_map: method_map,
        tcx: tcx,
    };
    visit::walk_crate(&mut v, krate, false);
    tcx.sess.abort_if_errors();
}

fn check_item(v: &mut CheckCrateVisitor, it: &Item, _is_const: bool) {
    match it.node {
        ItemStatic(_, _, ex) => {
            v.visit_expr(ex, true);
            check_item_recursion(&v.tcx.sess, &v.tcx.map, v.def_map, it);
        }
        ItemEnum(ref enum_definition, _) => {
            for var in (*enum_definition).variants.iter() {
                for ex in var.node.disr_expr.iter() {
                    v.visit_expr(*ex, true);
                }
            }
        }
        _ => visit::walk_item(v, it, false)
    }
}

fn check_pat(v: &mut CheckCrateVisitor, p: &Pat, _is_const: bool) {
    fn is_str(e: &Expr) -> bool {
        match e.node {
            ExprVstore(expr, ExprVstoreUniq) => {
                match expr.node {
                    ExprLit(lit) => ast_util::lit_is_str(lit),
                    _ => false,
                }
            }
            _ => false,
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

fn check_expr(v: &mut CheckCrateVisitor, e: &Expr, is_const: bool) {
    if is_const {
        match e.node {
          ExprUnary(UnDeref, _) => { }
          ExprUnary(UnBox, _) | ExprUnary(UnUniq, _) => {
            v.tcx.sess.span_err(e.span,
                                "cannot do allocations in constant expressions");
            return;
          }
          ExprLit(lit) if ast_util::lit_is_str(lit) => {}
          ExprBinary(..) | ExprUnary(..) => {
            let method_call = typeck::MethodCall::expr(e.id);
            if v.method_map.borrow().contains_key(&method_call) {
                v.tcx.sess.span_err(e.span, "user-defined operators are not \
                                             allowed in constant expressions");
            }
          }
          ExprLit(_) => (),
          ExprCast(_, _) => {
            let ety = ty::expr_ty(v.tcx, e);
            if !ty::type_is_numeric(ety) && !ty::type_is_unsafe_ptr(ety) {
                v.tcx.sess.span_err(e.span, ~"can not cast to `" +
                                              ppaux::ty_to_str(v.tcx, ety) +
                                             "` in a constant expression");
            }
          }
          ExprPath(ref pth) => {
            // NB: In the future you might wish to relax this slightly
            // to handle on-demand instantiation of functions via
            // foo::<bar> in a const. Currently that is only done on
            // a path in trans::callee that only works in block contexts.
            if !pth.segments.iter().all(|segment| segment.types.is_empty()) {
                v.tcx.sess.span_err(e.span,
                                    "paths in constants may only refer to \
                                     items without type parameters");
            }
            match v.def_map.borrow().find(&e.id) {
              Some(&DefStatic(..)) |
              Some(&DefFn(_, _)) |
              Some(&DefVariant(_, _, _)) |
              Some(&DefStruct(_)) => { }

              Some(&def) => {
                debug!("(checking const) found bad def: {:?}", def);
                v.tcx.sess.span_err(e.span,
                    "paths in constants may only refer to \
                     constants or functions");
              }
              None => {
                v.tcx.sess.span_bug(e.span, "unbound path in const?!");
              }
            }
          }
          ExprCall(callee, _) => {
            match v.def_map.borrow().find(&callee.id) {
                Some(&DefStruct(..)) => {}    // OK.
                Some(&DefVariant(..)) => {}    // OK.
                _ => {
                    v.tcx.sess.span_err(e.span,
                        "function calls in constants are limited to \
                         struct and enum constructors");
                }
            }
          }
          ExprVstore(_, ExprVstoreMutSlice) |
          ExprVstore(_, ExprVstoreSlice) |
          ExprVec(_) |
          ExprAddrOf(MutImmutable, _) |
          ExprParen(..) |
          ExprField(..) |
          ExprIndex(..) |
          ExprTup(..) |
          ExprRepeat(..) |
          ExprStruct(..) => { }
          ExprAddrOf(..) => {
                v.tcx.sess.span_err(e.span,
                    "references in constants may only refer to \
                     immutable values");
          },
          ExprVstore(_, ExprVstoreUniq) => {
              v.tcx.sess.span_err(e.span, "cannot allocate vectors in constant expressions")
          },

          _ => {
            v.tcx.sess.span_err(e.span,
                                "constant contains unimplemented expression type");
            return;
          }
        }
    }
    visit::walk_expr(v, e, is_const);
}

struct CheckItemRecursionVisitor<'a> {
    root_it: &'a Item,
    sess: &'a Session,
    ast_map: &'a ast_map::Map,
    def_map: resolve::DefMap,
    idstack: Vec<NodeId> }

// Make sure a const item doesn't recursively refer to itself
// FIXME: Should use the dependency graph when it's available (#1356)
pub fn check_item_recursion<'a>(sess: &'a Session,
                                ast_map: &'a ast_map::Map,
                                def_map: resolve::DefMap,
                                it: &'a Item) {

    let mut visitor = CheckItemRecursionVisitor {
        root_it: it,
        sess: sess,
        ast_map: ast_map,
        def_map: def_map,
        idstack: Vec::new()
    };
    visitor.visit_item(it, ());
}

impl<'a> Visitor<()> for CheckItemRecursionVisitor<'a> {
    fn visit_item(&mut self, it: &Item, _: ()) {
        if self.idstack.iter().any(|x| x == &(it.id)) {
            self.sess.span_fatal(self.root_it.span, "recursive constant");
        }
        self.idstack.push(it.id);
        visit::walk_item(self, it, ());
        self.idstack.pop();
    }

    fn visit_expr(&mut self, e: &Expr, _: ()) {
        match e.node {
            ExprPath(..) => {
                match self.def_map.borrow().find(&e.id) {
                    Some(&DefStatic(def_id, _)) if
                            ast_util::is_local(def_id) => {
                        self.visit_item(self.ast_map.expect_item(def_id.node), ());
                    }
                    _ => ()
                }
            },
            _ => ()
        }
        visit::walk_expr(self, e, ());
    }
}
