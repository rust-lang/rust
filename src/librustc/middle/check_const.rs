// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use middle::def::*;
use middle::ty;
use util::ppaux;

use syntax::ast;
use syntax::ast_util;
use syntax::visit::Visitor;
use syntax::visit;

struct CheckCrateVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    in_const: bool
}

impl<'a, 'tcx> CheckCrateVisitor<'a, 'tcx> {
    fn with_const<F>(&mut self, in_const: bool, f: F) where
        F: FnOnce(&mut CheckCrateVisitor<'a, 'tcx>),
    {
        let was_const = self.in_const;
        self.in_const = in_const;
        f(self);
        self.in_const = was_const;
    }
    fn inside_const<F>(&mut self, f: F) where
        F: FnOnce(&mut CheckCrateVisitor<'a, 'tcx>),
    {
        self.with_const(true, f);
    }
    fn outside_const<F>(&mut self, f: F) where
        F: FnOnce(&mut CheckCrateVisitor<'a, 'tcx>),
    {
        self.with_const(false, f);
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for CheckCrateVisitor<'a, 'tcx> {
    fn visit_item(&mut self, i: &ast::Item) {
        check_item(self, i);
    }
    fn visit_pat(&mut self, p: &ast::Pat) {
        check_pat(self, p);
    }
    fn visit_expr(&mut self, ex: &ast::Expr) {
        if check_expr(self, ex) {
            visit::walk_expr(self, ex);
        }
    }
}

pub fn check_crate(tcx: &ty::ctxt) {
    visit::walk_crate(&mut CheckCrateVisitor { tcx: tcx, in_const: false },
                      tcx.map.krate());
    tcx.sess.abort_if_errors();
}

fn check_item(v: &mut CheckCrateVisitor, it: &ast::Item) {
    match it.node {
        ast::ItemStatic(_, _, ref ex) |
        ast::ItemConst(_, ref ex) => {
            v.inside_const(|v| v.visit_expr(&**ex));
        }
        ast::ItemEnum(ref enum_definition, _) => {
            for var in (*enum_definition).variants.iter() {
                for ex in var.node.disr_expr.iter() {
                    v.inside_const(|v| v.visit_expr(&**ex));
                }
            }
        }
        _ => v.outside_const(|v| visit::walk_item(v, it))
    }
}

fn check_pat(v: &mut CheckCrateVisitor, p: &ast::Pat) {
    fn is_str(e: &ast::Expr) -> bool {
        match e.node {
            ast::ExprBox(_, ref expr) => {
                match expr.node {
                    ast::ExprLit(ref lit) => ast_util::lit_is_str(&**lit),
                    _ => false,
                }
            }
            _ => false,
        }
    }
    match p.node {
        // Let through plain ~-string literals here
        ast::PatLit(ref a) => if !is_str(&**a) { v.inside_const(|v| v.visit_expr(&**a)); },
        ast::PatRange(ref a, ref b) => {
            if !is_str(&**a) { v.inside_const(|v| v.visit_expr(&**a)); }
            if !is_str(&**b) { v.inside_const(|v| v.visit_expr(&**b)); }
        }
        _ => v.outside_const(|v| visit::walk_pat(v, p))
    }
}

fn check_expr(v: &mut CheckCrateVisitor, e: &ast::Expr) -> bool {
    if !v.in_const { return true }

    match e.node {
        ast::ExprUnary(ast::UnDeref, _) => {}
        ast::ExprUnary(ast::UnUniq, _) => {
            span_err!(v.tcx.sess, e.span, E0010,
                      "cannot do allocations in constant expressions");
            return false;
        }
        ast::ExprLit(ref lit) if ast_util::lit_is_str(&**lit) => {}
        ast::ExprBinary(..) | ast::ExprUnary(..) => {
            let method_call = ty::MethodCall::expr(e.id);
            if v.tcx.method_map.borrow().contains_key(&method_call) {
                span_err!(v.tcx.sess, e.span, E0011,
                          "user-defined operators are not allowed in constant \
                           expressions");
            }
        }
        ast::ExprLit(_) => (),
        ast::ExprCast(ref from, _) => {
            let toty = ty::expr_ty(v.tcx, e);
            let fromty = ty::expr_ty(v.tcx, &**from);
            let is_legal_cast =
                ty::type_is_numeric(toty) ||
                ty::type_is_unsafe_ptr(toty) ||
                (ty::type_is_bare_fn(toty) && ty::type_is_bare_fn_item(fromty));
            if !is_legal_cast {
                span_err!(v.tcx.sess, e.span, E0012,
                          "can not cast to `{}` in a constant expression",
                          ppaux::ty_to_string(v.tcx, toty));
            }
            if ty::type_is_unsafe_ptr(fromty) && ty::type_is_numeric(toty) {
                span_err!(v.tcx.sess, e.span, E0018,
                          "can not cast a pointer to an integer in a constant \
                           expression");
            }
        }
        ast::ExprPath(ref pth) => {
            // NB: In the future you might wish to relax this slightly
            // to handle on-demand instantiation of functions via
            // foo::<bar> in a const. Currently that is only done on
            // a path in trans::callee that only works in block contexts.
            if !pth.segments.iter().all(|segment| segment.parameters.is_empty()) {
                span_err!(v.tcx.sess, e.span, E0013,
                          "paths in constants may only refer to items without \
                           type parameters");
            }
            match v.tcx.def_map.borrow().get(&e.id) {
                Some(&DefStatic(..)) |
                Some(&DefConst(..)) |
                Some(&DefFn(..)) |
                Some(&DefVariant(_, _, _)) |
                Some(&DefStruct(_)) => { }

                Some(&def) => {
                    debug!("(checking const) found bad def: {}", def);
                    span_err!(v.tcx.sess, e.span, E0014,
                              "paths in constants may only refer to constants \
                               or functions");
                }
                None => {
                    v.tcx.sess.span_bug(e.span, "unbound path in const?!");
                }
            }
        }
        ast::ExprCall(ref callee, _) => {
            match v.tcx.def_map.borrow().get(&callee.id) {
                Some(&DefStruct(..)) |
                Some(&DefVariant(..)) => {}    // OK.

                _ => {
                    span_err!(v.tcx.sess, e.span, E0015,
                              "function calls in constants are limited to \
                               struct and enum constructors");
                }
            }
        }
        ast::ExprBlock(ref block) => {
            // Check all statements in the block
            for stmt in block.stmts.iter() {
                let block_span_err = |span|
                    span_err!(v.tcx.sess, span, E0016,
                              "blocks in constants are limited to items and \
                               tail expressions");
                match stmt.node {
                    ast::StmtDecl(ref span, _) => {
                        match span.node {
                            ast::DeclLocal(_) => block_span_err(span.span),

                            // Item statements are allowed
                            ast::DeclItem(_) => {}
                        }
                    }
                    ast::StmtExpr(ref expr, _) => block_span_err(expr.span),
                    ast::StmtSemi(ref semi, _) => block_span_err(semi.span),
                    ast::StmtMac(..) => {
                        v.tcx.sess.span_bug(e.span, "unexpanded statement \
                                                     macro in const?!")
                    }
                }
            }
            match block.expr {
                Some(ref expr) => { check_expr(v, &**expr); }
                None => {}
            }
        }
        ast::ExprVec(_) |
        ast::ExprAddrOf(ast::MutImmutable, _) |
        ast::ExprParen(..) |
        ast::ExprField(..) |
        ast::ExprTupField(..) |
        ast::ExprIndex(..) |
        ast::ExprTup(..) |
        ast::ExprRepeat(..) |
        ast::ExprStruct(..) => {}

        ast::ExprAddrOf(_, ref inner) => {
            match inner.node {
                // Mutable slices are allowed.
                ast::ExprVec(_) => {}
                _ => span_err!(v.tcx.sess, e.span, E0017,
                               "references in constants may only refer \
                                to immutable values")

            }
        }

        _ => {
            span_err!(v.tcx.sess, e.span, E0019,
                      "constant contains unimplemented expression type");
            return false;
        }
    }
    true
}
