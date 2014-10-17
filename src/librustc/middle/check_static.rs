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
use middle::def;
use middle::typeck;
use middle::traits;
use middle::mem_categorization as mc;
use middle::expr_use_visitor as euv;
use util::nodemap::NodeSet;

use syntax::ast;
use syntax::print::pprust;
use syntax::visit::Visitor;
use syntax::codemap::{DUMMY_SP, Span};
use syntax::visit;

#[deriving(Eq, PartialEq)]
enum Mode {
    InConstant,
    InStatic,
    InStaticMut,
    InNothing,
}

struct CheckStaticVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    mode: Mode,
    checker: &'a mut GlobalChecker,
}

struct GlobalVisitor<'a, 'b, 't: 'b>(euv::ExprUseVisitor<'a, 'b, ty::ctxt<'t>>);
struct GlobalChecker {
    static_consumptions: NodeSet,
    const_borrows: NodeSet,
    static_interior_borrows: NodeSet,
}

pub fn check_crate(tcx: &ty::ctxt) {
    let mut checker = GlobalChecker {
        static_consumptions: NodeSet::new(),
        const_borrows: NodeSet::new(),
        static_interior_borrows: NodeSet::new(),
    };
    {
        let visitor = euv::ExprUseVisitor::new(&mut checker, tcx);
        visit::walk_crate(&mut GlobalVisitor(visitor), tcx.map.krate());
    }
    visit::walk_crate(&mut CheckStaticVisitor {
        tcx: tcx,
        mode: InNothing,
        checker: &mut checker,
    }, tcx.map.krate());
}

impl<'a, 'tcx> CheckStaticVisitor<'a, 'tcx> {
    fn with_mode(&mut self, mode: Mode, f: |&mut CheckStaticVisitor<'a, 'tcx>|) {
        let old = self.mode;
        self.mode = mode;
        f(self);
        self.mode = old;
    }

    fn msg(&self) -> &'static str {
        match self.mode {
            InConstant => "constants",
            InStaticMut | InStatic => "statics",
            InNothing => unreachable!(),
        }
    }

    fn check_static_mut_type(&self, e: &ast::Expr) {
        let node_ty = ty::node_id_to_type(self.tcx, e.id);
        let tcontents = ty::type_contents(self.tcx, node_ty);

        let suffix = if tcontents.has_dtor() {
            "destructors"
        } else if tcontents.owns_owned() {
            "owned pointers"
        } else {
            return
        };

        self.tcx.sess.span_err(e.span, format!("mutable statics are not allowed \
                                                to have {}", suffix).as_slice());
    }

    fn check_static_type(&self, e: &ast::Expr) {
        let ty = ty::node_id_to_type(self.tcx, e.id);
        let infcx = typeck::infer::new_infer_ctxt(self.tcx);
        let mut fulfill_cx = traits::FulfillmentContext::new();
        let cause = traits::ObligationCause::misc(DUMMY_SP);
        let obligation = traits::obligation_for_builtin_bound(self.tcx, cause, ty,
                                                              ty::BoundSync);
        fulfill_cx.register_obligation(self.tcx, obligation.unwrap());
        let env = ty::empty_parameter_environment();
        let result = fulfill_cx.select_all_or_error(&infcx, &env, self.tcx).is_ok();
        if !result {
            self.tcx.sess.span_err(e.span, "shared static items must have a \
                                            type which implements Sync");
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for CheckStaticVisitor<'a, 'tcx> {
    fn visit_item(&mut self, i: &ast::Item) {
        debug!("visit_item(item={})", pprust::item_to_string(i));
        match i.node {
            ast::ItemStatic(_, ast::MutImmutable, ref expr) => {
                self.check_static_type(&**expr);
                self.with_mode(InStatic, |v| v.visit_expr(&**expr));
            }
            ast::ItemStatic(_, ast::MutMutable, ref expr) => {
                self.check_static_mut_type(&**expr);
                self.with_mode(InStaticMut, |v| v.visit_expr(&**expr));
            }
            ast::ItemConst(_, ref expr) => {
                self.with_mode(InConstant, |v| v.visit_expr(&**expr));
            }
            _ => {
                self.with_mode(InNothing, |v| visit::walk_item(v, i));
            }
        }
    }

    /// This method is used to enforce the constraints on
    /// immutable static items. It walks through the *value*
    /// of the item walking down the expression and evaluating
    /// every nested expression. if the expression is not part
    /// of a static item, this method does nothing but walking
    /// down through it.
    fn visit_expr(&mut self, e: &ast::Expr) {
        if self.mode == InNothing {
            return visit::walk_expr(self, e);
        }

        let node_ty = ty::node_id_to_type(self.tcx, e.id);

        match ty::get(node_ty).sty {
            ty::ty_struct(did, _) |
            ty::ty_enum(did, _) if ty::has_dtor(self.tcx, did) => {
                self.tcx.sess.span_err(e.span,
                                       format!("{} are not allowed to have \
                                                destructors", self.msg()).as_slice())
            }
            _ => {}
        }

        // statics cannot be consumed by value at any time, that would imply
        // that they're an initializer (what a const is for) or kept in sync
        // over time (not feasible), so deny it outright.
        if self.checker.static_consumptions.remove(&e.id) {
            self.tcx.sess.span_err(e.span, "cannot refer to other statics by \
                                            value, use the address-of operator \
                                            or a constant instead");
        }

        // Borrowed statics can specifically *only* have their address taken,
        // not any number of other borrows such as borrowing fields, reading
        // elements of an array, etc.
        if self.checker.static_interior_borrows.remove(&e.id) {
            self.tcx.sess.span_err(e.span, "cannot refer to the interior of \
                                            another static, use a constant \
                                            instead");
        }

        // constants cannot be borrowed if they contain interior mutability as
        // it means that our "silent insertion of statics" could change
        // initializer values (very bad).
        if self.checker.const_borrows.remove(&e.id) {
            let node_ty = ty::node_id_to_type(self.tcx, e.id);
            let tcontents = ty::type_contents(self.tcx, node_ty);
            if tcontents.interior_unsafe() {
                self.tcx.sess.span_err(e.span, "cannot borrow a constant which \
                                                contains interior mutability, \
                                                create a static instead");
            }
        }

        match e.node {
            ast::ExprAddrOf(ast::MutMutable, _) => {
                if self.mode != InStaticMut {
                    span_err!(self.tcx.sess, e.span, E0020,
                              "{} are not allowed to have mutable references",
                              self.msg());
                }
            },
            ast::ExprBox(..) |
            ast::ExprUnary(ast::UnUniq, _) => {
                span_err!(self.tcx.sess, e.span, E0022,
                          "{} are not allowed to have custom pointers",
                          self.msg());
            }
            ast::ExprPath(..) => {
                match ty::resolve_expr(self.tcx, e) {
                    def::DefStatic(..) if self.mode == InConstant => {
                        let msg = "constants cannot refer to other statics, \
                                   insert an intermediate constant \
                                   instead";
                        self.tcx.sess.span_err(e.span, msg.as_slice());
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        visit::walk_expr(self, e);
    }
}

impl<'a, 'b, 't, 'v> Visitor<'v> for GlobalVisitor<'a, 'b, 't> {
    fn visit_item(&mut self, item: &ast::Item) {
        match item.node {
            ast::ItemConst(_, ref e) |
            ast::ItemStatic(_, _, ref e) => {
                let GlobalVisitor(ref mut v) = *self;
                v.consume_expr(&**e);
            }
            _ => {}
        }
        visit::walk_item(self, item);
    }
}

impl euv::Delegate for GlobalChecker {
    fn consume(&mut self,
               consume_id: ast::NodeId,
               _consume_span: Span,
               cmt: mc::cmt,
               _mode: euv::ConsumeMode) {
        let mut cur = &cmt;
        loop {
            match cur.cat {
                mc::cat_static_item => {
                    self.static_consumptions.insert(consume_id);
                    break
                }
                mc::cat_deref(ref cmt, _, _) |
                mc::cat_discr(ref cmt, _) |
                mc::cat_downcast(ref cmt) |
                mc::cat_interior(ref cmt, _) => cur = cmt,

                mc::cat_rvalue(..) |
                mc::cat_upvar(..) |
                mc::cat_local(..) => break,
            }
        }
    }
    fn borrow(&mut self,
              borrow_id: ast::NodeId,
              _borrow_span: Span,
              cmt: mc::cmt,
              _loan_region: ty::Region,
              _bk: ty::BorrowKind,
              _loan_cause: euv::LoanCause) {
        let mut cur = &cmt;
        let mut is_interior = false;
        loop {
            match cur.cat {
                mc::cat_rvalue(..) => {
                    self.const_borrows.insert(borrow_id);
                    break
                }
                mc::cat_static_item => {
                    if is_interior {
                        self.static_interior_borrows.insert(borrow_id);
                    }
                    break
                }
                mc::cat_deref(ref cmt, _, _) |
                mc::cat_interior(ref cmt, _) => {
                    is_interior = true;
                    cur = cmt;
                }

                mc::cat_downcast(..) |
                mc::cat_discr(..) |
                mc::cat_upvar(..) |
                mc::cat_local(..) => unreachable!(),
            }
        }
    }

    fn decl_without_init(&mut self,
                         _id: ast::NodeId,
                         _span: Span) {}
    fn mutate(&mut self,
              _assignment_id: ast::NodeId,
              _assignment_span: Span,
              _assignee_cmt: mc::cmt,
              _mode: euv::MutateMode) {}
    fn consume_pat(&mut self,
                   _consume_pat: &ast::Pat,
                   _cmt: mc::cmt,
                   _mode: euv::ConsumeMode) {}
}

