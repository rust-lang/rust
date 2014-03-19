// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Verifies that the types and values of static items
// are safe. The rules enforced by this module are:
//
// - For each *mutable* static item, it checks that its **type**:
//     - doesn't have a destructor
//     - doesn't own an owned pointer
//
// - For each *immutable* static item, it checks that its **value**:
//       - doesn't own owned, managed pointers
//       - doesn't contain a struct literal or a call to an enum variant / struct constructor where
//           - the type of the struct/enum is not freeze
//           - the type of the struct/enum has a dtor

use middle::ty;

use syntax::ast;
use syntax::codemap::Span;
use syntax::visit::Visitor;
use syntax::visit;
use syntax::print::pprust;


fn safe_type_for_static_mut(cx: &ty::ctxt, e: &ast::Expr) -> Option<~str> {
    let node_ty = ty::node_id_to_type(cx, e.id);
    let tcontents = ty::type_contents(cx, node_ty);
    debug!("safe_type_for_static_mut(dtor={}, managed={}, owned={})",
           tcontents.has_dtor(), tcontents.owns_managed(), tcontents.owns_owned())

    let suffix = if tcontents.has_dtor() {
        "destructors"
    } else if tcontents.owns_managed() {
        "managed pointers"
    } else if tcontents.owns_owned() {
        "owned pointers"
    } else {
        return None;
    };

    Some(format!("mutable static items are not allowed to have {}", suffix))
}

struct CheckStaticVisitor<'a> {
    tcx: &'a ty::ctxt,
}

pub fn check_crate(tcx: &ty::ctxt, krate: &ast::Crate) {
    visit::walk_crate(&mut CheckStaticVisitor { tcx: tcx }, krate, false)
}

impl<'a> CheckStaticVisitor<'a> {
    fn report_error(&self, span: Span, result: Option<~str>) -> bool {
        match result {
            None => { false }
            Some(msg) => {
                self.tcx.sess.span_err(span, msg);
                true
            }
        }
    }
}

impl<'a> Visitor<bool> for CheckStaticVisitor<'a> {

    fn visit_item(&mut self, i: &ast::Item, _is_const: bool) {
        debug!("visit_item(item={})", pprust::item_to_str(i));
        match i.node {
            ast::ItemStatic(_, mutability, expr) => {
                match mutability {
                    ast::MutImmutable => {
                        self.visit_expr(expr, true);
                    }
                    ast::MutMutable => {
                        self.report_error(expr.span, safe_type_for_static_mut(self.tcx, expr));
                    }
                }
            }
            _ => { visit::walk_item(self, i, false) }
        }
    }

    /// This method is used to enforce the constraints on
    /// immutable static items. It walks through the *value*
    /// of the item walking down the expression and evaluating
    /// every nested expression. if the expression is not part
    /// of a static item, this method does nothing but walking
    /// down through it.
    fn visit_expr(&mut self, e: &ast::Expr, is_const: bool) {
        debug!("visit_expr(expr={})", pprust::expr_to_str(e));

        if !is_const {
            return visit::walk_expr(self, e, is_const);
        }

        match e.node {
            ast::ExprField(..) | ast::ExprVec(..) |
            ast::ExprBlock(..) | ast::ExprTup(..) |
            ast::ExprVstore(_, ast::ExprVstoreSlice) => {
                visit::walk_expr(self, e, is_const);
            }
            ast::ExprVstore(_, ast::ExprVstoreMutSlice) => {
                self.tcx.sess.span_err(e.span,
                                       "static items are not allowed to have mutable slices");
           },
            ast::ExprUnary(ast::UnBox, _) => {
                self.tcx.sess.span_err(e.span,
                                   "static items are not allowed to have managed pointers");
            }
            ast::ExprBox(..) |
            ast::ExprUnary(ast::UnUniq, _) |
            ast::ExprVstore(_, ast::ExprVstoreUniq) => {
                self.tcx.sess.span_err(e.span,
                                   "static items are not allowed to have owned pointers");
            }
            ast::ExprProc(..) => {
                self.report_error(e.span,
                                  Some(~"immutable static items must be `Freeze`"));
                return;
            }
            ast::ExprAddrOf(mutability, _) => {
                match mutability {
                    ast::MutMutable => {
                        self.report_error(e.span,
                                  Some(~"immutable static items must be `Freeze`"));
                        return;
                    }
                    _ => {}
                }
            }
            _ => {
                let node_ty = ty::node_id_to_type(self.tcx, e.id);

                match ty::get(node_ty).sty {
                    ty::ty_struct(did, _) |
                    ty::ty_enum(did, _) => {
                        if ty::has_dtor(self.tcx, did) {
                            self.report_error(e.span,
                                     Some(~"static items are not allowed to have destructors"));
                            return;
                        }
                        if Some(did) == self.tcx.lang_items.no_freeze_bound() {
                            self.report_error(e.span,
                                              Some(~"immutable static items must be `Freeze`"));
                            return;
                        }
                    }
                    _ => {}
                }
                visit::walk_expr(self, e, is_const);
            }
        }
    }
}
