// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Verifies that the types and values of const and static items
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

use self::Mode::*;

use middle::def;
use middle::expr_use_visitor as euv;
use middle::infer;
use middle::mem_categorization as mc;
use middle::traits;
use middle::ty;
use util::ppaux;

use syntax::ast;
use syntax::codemap::Span;
use syntax::print::pprust;
use syntax::visit::{self, Visitor};

#[derive(Copy, Eq, PartialEq)]
enum Mode {
    InConstant,
    InStatic,
    InStaticMut,
    InNothing,
}

struct CheckCrateVisitor<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    mode: Mode,
}

impl<'a, 'tcx> CheckCrateVisitor<'a, 'tcx> {
    fn with_mode<F>(&mut self, mode: Mode, f: F) where
        F: FnOnce(&mut CheckCrateVisitor<'a, 'tcx>),
    {
        let old = self.mode;
        self.mode = mode;
        f(self);
        self.mode = old;
    }

    fn msg(&self) -> &'static str {
        match self.mode {
            InConstant => "constant",
            InStaticMut | InStatic => "static",
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

        self.tcx.sess.span_err(e.span, &format!("mutable statics are not allowed \
                                                 to have {}", suffix)[]);
    }

    fn check_static_type(&self, e: &ast::Expr) {
        let ty = ty::node_id_to_type(self.tcx, e.id);
        let infcx = infer::new_infer_ctxt(self.tcx);
        let mut fulfill_cx = traits::FulfillmentContext::new();
        let cause = traits::ObligationCause::new(e.span, e.id, traits::SharedStatic);
        fulfill_cx.register_builtin_bound(&infcx, ty, ty::BoundSync, cause);
        let env = ty::empty_parameter_environment(self.tcx);
        match fulfill_cx.select_all_or_error(&infcx, &env) {
            Ok(()) => { },
            Err(ref errors) => {
                traits::report_fulfillment_errors(&infcx, errors);
            }
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for CheckCrateVisitor<'a, 'tcx> {
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
            ast::ItemEnum(ref enum_definition, _) => {
                self.with_mode(InConstant, |v| {
                    for var in &enum_definition.variants {
                        if let Some(ref ex) = var.node.disr_expr {
                            v.visit_expr(&**ex);
                        }
                    }
                });
            }
            _ => {
                self.with_mode(InNothing, |v| visit::walk_item(v, i));
            }
        }
    }

    fn visit_pat(&mut self, p: &ast::Pat) {
        let mode = match p.node {
            ast::PatLit(_) | ast::PatRange(..) => InConstant,
            _ => InNothing
        };
        self.with_mode(mode, |v| visit::walk_pat(v, p))
    }

    fn visit_expr(&mut self, ex: &ast::Expr) {
        if self.mode != InNothing {
            check_expr(self, ex);
        }
        visit::walk_expr(self, ex);
    }
}

/// This function is used to enforce the constraints on
/// const/static items. It walks through the *value*
/// of the item walking down the expression and evaluating
/// every nested expression. If the expression is not part
/// of a const/static item, this function does nothing but
/// walking down through it.
fn check_expr(v: &mut CheckCrateVisitor, e: &ast::Expr) {
    let node_ty = ty::node_id_to_type(v.tcx, e.id);

    match node_ty.sty {
        ty::ty_struct(did, _) |
        ty::ty_enum(did, _) if ty::has_dtor(v.tcx, did) => {
            v.tcx.sess.span_err(e.span,
                                &format!("{}s are not allowed to have destructors",
                                         v.msg())[])
        }
        _ => {}
    }

    match e.node {
        ast::ExprBox(..) |
        ast::ExprUnary(ast::UnUniq, _) => {
            span_err!(v.tcx.sess, e.span, E0010,
                      "allocations are not allowed in {}s", v.msg());
        }
        ast::ExprBinary(..) | ast::ExprUnary(..) => {
            let method_call = ty::MethodCall::expr(e.id);
            if v.tcx.method_map.borrow().contains_key(&method_call) {
                span_err!(v.tcx.sess, e.span, E0011,
                          "user-defined operators are not allowed in {}s", v.msg());
            }
        }
        ast::ExprCast(ref from, _) => {
            let toty = ty::expr_ty(v.tcx, e);
            let fromty = ty::expr_ty(v.tcx, &**from);
            let is_legal_cast =
                ty::type_is_numeric(toty) ||
                ty::type_is_unsafe_ptr(toty) ||
                (ty::type_is_bare_fn(toty) && ty::type_is_bare_fn_item(fromty));
            if !is_legal_cast {
                span_err!(v.tcx.sess, e.span, E0012,
                          "can not cast to `{}` in {}s",
                          ppaux::ty_to_string(v.tcx, toty), v.msg());
            }
            if ty::type_is_unsafe_ptr(fromty) && ty::type_is_numeric(toty) {
                span_err!(v.tcx.sess, e.span, E0018,
                          "can not cast a pointer to an integer in {}s", v.msg());
            }
        }
        ast::ExprPath(_) | ast::ExprQPath(_) => {
            match v.tcx.def_map.borrow()[e.id] {
                def::DefStatic(..) if v.mode == InConstant => {
                    span_err!(v.tcx.sess, e.span, E0013,
                              "constants cannot refer to other statics, \
                               insert an intermediate constant instead");
                }
                def::DefStatic(..) | def::DefConst(..) |
                def::DefFn(..) | def::DefStaticMethod(..) | def::DefMethod(..) |
                def::DefStruct(_) | def::DefVariant(_, _, _) => {}

                def => {
                    debug!("(checking const) found bad def: {:?}", def);
                    span_err!(v.tcx.sess, e.span, E0014,
                              "paths in constants may only refer to constants \
                               or functions");
                }
            }
        }
        ast::ExprCall(ref callee, _) => {
            match v.tcx.def_map.borrow()[callee.id] {
                def::DefStruct(..) | def::DefVariant(..) => {}    // OK.
                _ => {
                    span_err!(v.tcx.sess, e.span, E0015,
                              "function calls in constants are limited to \
                               struct and enum constructors");
                }
            }
        }
        ast::ExprBlock(ref block) => {
            // Check all statements in the block
            for stmt in &block.stmts {
                let block_span_err = |span|
                    span_err!(v.tcx.sess, span, E0016,
                              "blocks in constants are limited to items and \
                               tail expressions");
                match stmt.node {
                    ast::StmtDecl(ref decl, _) => {
                        match decl.node {
                            ast::DeclLocal(_) => block_span_err(decl.span),

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
        }
        ast::ExprAddrOf(ast::MutMutable, ref inner) => {
            match inner.node {
                // Mutable slices are allowed. Only in `static mut`.
                ast::ExprVec(_) if v.mode == InStaticMut => {}
                _ => span_err!(v.tcx.sess, e.span, E0017,
                               "references in {}s may only refer \
                                to immutable values", v.msg())
            }
        }

        ast::ExprLit(_) |
        ast::ExprVec(_) |
        ast::ExprAddrOf(ast::MutImmutable, _) |
        ast::ExprParen(..) |
        ast::ExprField(..) |
        ast::ExprTupField(..) |
        ast::ExprIndex(..) |
        ast::ExprTup(..) |
        ast::ExprRepeat(..) |
        ast::ExprStruct(..) => {}

        // Conditional control flow (possible to implement).
        ast::ExprMatch(..) |
        ast::ExprIf(..) |
        ast::ExprIfLet(..) |

        // Loops (not very meaningful in constants).
        ast::ExprWhile(..) |
        ast::ExprWhileLet(..) |
        ast::ExprForLoop(..) |
        ast::ExprLoop(..) |

        // More control flow (also not very meaningful).
        ast::ExprBreak(_) |
        ast::ExprAgain(_) |
        ast::ExprRet(_) |

        // Miscellaneous expressions that could be implemented.
        ast::ExprClosure(..) |
        ast::ExprRange(..) |

        // Various other expressions.
        ast::ExprMethodCall(..) |
        ast::ExprAssign(..) |
        ast::ExprAssignOp(..) |
        ast::ExprInlineAsm(_) |
        ast::ExprMac(_) => {
            span_err!(v.tcx.sess, e.span, E0019,
                      "{} contains unimplemented expression type", v.msg());
        }
    }
}

struct GlobalVisitor<'a,'b,'tcx:'a+'b>(
    euv::ExprUseVisitor<'a,'b,'tcx,ty::ParameterEnvironment<'b,'tcx>>);

struct GlobalChecker<'a,'tcx:'a> {
    tcx: &'a ty::ctxt<'tcx>
}

pub fn check_crate(tcx: &ty::ctxt) {
    let param_env = ty::empty_parameter_environment(tcx);
    let mut checker = GlobalChecker {
        tcx: tcx
    };
    let visitor = euv::ExprUseVisitor::new(&mut checker, &param_env);
    visit::walk_crate(&mut GlobalVisitor(visitor), tcx.map.krate());

    visit::walk_crate(&mut CheckCrateVisitor {
        tcx: tcx,
        mode: InNothing,
    }, tcx.map.krate());

    tcx.sess.abort_if_errors();
}

impl<'a,'b,'t,'v> Visitor<'v> for GlobalVisitor<'a,'b,'t> {
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

impl<'a, 'tcx> euv::Delegate<'tcx> for GlobalChecker<'a, 'tcx> {
    fn consume(&mut self,
               _consume_id: ast::NodeId,
               consume_span: Span,
               cmt: mc::cmt,
               _mode: euv::ConsumeMode) {
        let mut cur = &cmt;
        loop {
            match cur.cat {
                mc::cat_static_item => {
                    // statics cannot be consumed by value at any time, that would imply
                    // that they're an initializer (what a const is for) or kept in sync
                    // over time (not feasible), so deny it outright.
                    self.tcx.sess.span_err(consume_span,
                        "cannot refer to other statics by value, use the \
                         address-of operator or a constant instead");
                    break;
                }
                mc::cat_deref(ref cmt, _, _) |
                mc::cat_downcast(ref cmt, _) |
                mc::cat_interior(ref cmt, _) => cur = cmt,

                mc::cat_rvalue(..) |
                mc::cat_upvar(..) |
                mc::cat_local(..) => break
            }
        }
    }
    fn borrow(&mut self,
              _borrow_id: ast::NodeId,
              borrow_span: Span,
              cmt: mc::cmt<'tcx>,
              _loan_region: ty::Region,
              _bk: ty::BorrowKind,
              _loan_cause: euv::LoanCause) {
        let mut cur = &cmt;
        let mut is_interior = false;
        loop {
            match cur.cat {
                mc::cat_rvalue(..) => {
                    // constants cannot be borrowed if they contain interior mutability as
                    // it means that our "silent insertion of statics" could change
                    // initializer values (very bad).
                    if ty::type_contents(self.tcx, cur.ty).interior_unsafe() {
                        self.tcx.sess.span_err(borrow_span,
                            "cannot borrow a constant which contains \
                            interior mutability, create a static instead");
                    }
                    break;
                }
                mc::cat_static_item => {
                    if is_interior {
                        // Borrowed statics can specifically *only* have their address taken,
                        // not any number of other borrows such as borrowing fields, reading
                        // elements of an array, etc.
                        self.tcx.sess.span_err(borrow_span,
                            "cannot refer to the interior of another \
                             static, use a constant instead");
                    }
                    break;
                }
                mc::cat_deref(ref cmt, _, _) |
                mc::cat_downcast(ref cmt, _) |
                mc::cat_interior(ref cmt, _) => {
                    is_interior = true;
                    cur = cmt;
                }

                mc::cat_upvar(..) |
                mc::cat_local(..) => break
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

    fn matched_pat(&mut self,
                   _: &ast::Pat,
                   _: mc::cmt,
                   _: euv::MatchMode) {}

    fn consume_pat(&mut self,
                   _consume_pat: &ast::Pat,
                   _cmt: mc::cmt,
                   _mode: euv::ConsumeMode) {}
}