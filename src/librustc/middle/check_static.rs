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
//           - the type of the struct/enum has a dtor
//
// Rules Enforced Elsewhere:
// - It's not possible to take the address of a static item with unsafe interior. This is enforced
// by borrowck::gather_loans

use middle::ty;

use syntax::ast;
use syntax::visit::Visitor;
use syntax::visit;
use syntax::print::pprust;


fn safe_type_for_static_mut(cx: &ty::ctxt, e: &ast::Expr) -> Option<String> {
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

struct CheckStaticVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    in_const: bool
}

pub fn check_crate(tcx: &ty::ctxt) {
    visit::walk_crate(&mut CheckStaticVisitor { tcx: tcx, in_const: false },
                      tcx.map.krate())
}

impl<'a, 'tcx> CheckStaticVisitor<'a, 'tcx> {
    fn with_const(&mut self, in_const: bool, f: |&mut CheckStaticVisitor<'a, 'tcx>|) {
        let was_const = self.in_const;
        self.in_const = in_const;
        f(self);
        self.in_const = was_const;
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for CheckStaticVisitor<'a, 'tcx> {
    fn visit_item(&mut self, i: &ast::Item) {
        debug!("visit_item(item={})", pprust::item_to_string(i));
        match i.node {
            ast::ItemStatic(_, mutability, ref expr) => {
                match mutability {
                    ast::MutImmutable => {
                        self.with_const(true, |v| v.visit_expr(&**expr));
                    }
                    ast::MutMutable => {
                        match safe_type_for_static_mut(self.tcx, &**expr) {
                            Some(msg) => {
                                self.tcx.sess.span_err(expr.span, msg.as_slice());
                            }
                            None => {}
                        }
                    }
                }
            }
            _ => self.with_const(false, |v| visit::walk_item(v, i))
        }
    }

    /// This method is used to enforce the constraints on
    /// immutable static items. It walks through the *value*
    /// of the item walking down the expression and evaluating
    /// every nested expression. if the expression is not part
    /// of a static item, this method does nothing but walking
    /// down through it.
    fn visit_expr(&mut self, e: &ast::Expr) {
        debug!("visit_expr(expr={})", pprust::expr_to_string(e));

        if !self.in_const {
            return visit::walk_expr(self, e);
        }

        match e.node {
            ast::ExprField(..) | ast::ExprTupField(..) | ast::ExprVec(..) |
            ast::ExprBlock(..) | ast::ExprTup(..)  => {
                visit::walk_expr(self, e);
            }
            ast::ExprAddrOf(ast::MutMutable, _) => {
                span_err!(self.tcx.sess, e.span, E0020,
                    "static items are not allowed to have mutable slices");
            },
            ast::ExprUnary(ast::UnBox, _) => {
                span_err!(self.tcx.sess, e.span, E0021,
                    "static items are not allowed to have managed pointers");
            }
            ast::ExprBox(..) |
            ast::ExprUnary(ast::UnUniq, _) => {
                span_err!(self.tcx.sess, e.span, E0022,
                    "static items are not allowed to have custom pointers");
            }
            _ => {
                let node_ty = ty::node_id_to_type(self.tcx, e.id);

                match ty::get(node_ty).sty {
                    ty::ty_struct(did, _) |
                    ty::ty_enum(did, _) => {
                        if ty::has_dtor(self.tcx, did) {
                            self.tcx.sess.span_err(e.span,
                                "static items are not allowed to have destructors");
                            return;
                        }
                    }
                    _ => {}
                }
                visit::walk_expr(self, e);
            }
        }
    }
}
