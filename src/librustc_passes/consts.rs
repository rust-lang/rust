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
//     - doesn't own a box
//
// - For each *immutable* static item, it checks that its **value**:
//       - doesn't own a box
//       - doesn't contain a struct literal or a call to an enum variant / struct constructor where
//           - the type of the struct/enum has a dtor
//
// Rules Enforced Elsewhere:
// - It's not possible to take the address of a static item with unsafe interior. This is enforced
// by borrowck::gather_loans

use rustc::dep_graph::DepNode;
use rustc::ty::cast::CastKind;
use rustc_const_eval::{ConstEvalErr, ConstContext};
use rustc_const_eval::ErrKind::{IndexOpFeatureGated, UnimplementedConstVal, MiscCatchAll, Math};
use rustc_const_eval::ErrKind::{ErroneousReferencedConstant, MiscBinaryOp, NonConstPath, BadType};
use rustc_const_eval::ErrKind::UnresolvedPath;
use rustc_const_eval::EvalHint::ExprTypeChecked;
use rustc_const_math::{ConstMathErr, Op};
use rustc::hir::def::{Def, CtorKind};
use rustc::hir::def_id::DefId;
use rustc::hir::map::blocks::FnLikeNode;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::Categorization;
use rustc::mir::transform::MirSource;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::traits::Reveal;
use rustc::util::common::ErrorReported;
use rustc::util::nodemap::NodeSet;
use rustc::lint::builtin::CONST_ERR;

use rustc::hir::{self, PatKind};
use syntax::ast;
use syntax_pos::Span;
use rustc::hir::intravisit::{self, Visitor, NestedVisitorMap};

use std::collections::hash_map::Entry;
use std::cmp::Ordering;
use std::mem;

struct CheckCrateVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    in_fn: bool,
    promotable: bool,
    mut_rvalue_borrows: NodeSet,
    param_env: ty::ParameterEnvironment<'tcx>,
    tables: &'a ty::Tables<'tcx>,
}

impl<'a, 'gcx> CheckCrateVisitor<'a, 'gcx> {
    fn check_const_eval(&self, expr: &'gcx hir::Expr) {
        let const_cx = ConstContext::with_tables(self.tcx, self.tables);
        if let Err(err) = const_cx.eval(expr, ExprTypeChecked) {
            match err.kind {
                UnimplementedConstVal(_) => {}
                IndexOpFeatureGated => {}
                ErroneousReferencedConstant(_) => {}
                BadType(_) => {}
                _ => {
                    self.tcx.sess.add_lint(CONST_ERR,
                                           expr.id,
                                           expr.span,
                                           format!("constant evaluation error: {}. This will \
                                                    become a HARD ERROR in the future",
                                                   err.description().into_oneline()))
                }
            }
        }
    }

    // Adds the worst effect out of all the values of one type.
    fn add_type(&mut self, ty: Ty<'gcx>) {
        if ty.type_contents(self.tcx).interior_unsafe() {
            self.promotable = false;
        }

        if self.tcx.type_needs_drop_given_env(ty, &self.param_env) {
            self.promotable = false;
        }
    }

    fn handle_const_fn_call(&mut self, def_id: DefId, ret_ty: Ty<'gcx>) {
        self.add_type(ret_ty);

        self.promotable &= if let Some(fn_id) = self.tcx.map.as_local_node_id(def_id) {
            FnLikeNode::from_node(self.tcx.map.get(fn_id)).map_or(false, |fn_like| {
                fn_like.constness() == hir::Constness::Const
            })
        } else {
            self.tcx.sess.cstore.is_const_fn(def_id)
        };
    }
}

impl<'a, 'tcx> Visitor<'tcx> for CheckCrateVisitor<'a, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }

    fn visit_nested_body(&mut self, body_id: hir::BodyId) {
        match self.tcx.rvalue_promotable_to_static.borrow_mut().entry(body_id.node_id) {
            Entry::Occupied(_) => return,
            Entry::Vacant(entry) => {
                // Prevent infinite recursion on re-entry.
                entry.insert(false);
            }
        }

        let item_id = self.tcx.map.body_owner(body_id);

        let outer_in_fn = self.in_fn;
        self.in_fn = match MirSource::from_node(self.tcx, item_id) {
            MirSource::Fn(_) => true,
            _ => false
        };

        let outer_tables = self.tables;
        self.tables = self.tcx.item_tables(self.tcx.map.local_def_id(item_id));

        let body = self.tcx.map.body(body_id);
        if !self.in_fn {
            self.check_const_eval(&body.value);
        }

        let outer_penv = self.tcx.infer_ctxt(body_id, Reveal::NotSpecializable).enter(|infcx| {
            let param_env = infcx.parameter_environment.clone();
            let outer_penv = mem::replace(&mut self.param_env, param_env);
            euv::ExprUseVisitor::new(self, &infcx).consume_body(body);
            outer_penv
        });

        self.visit_body(body);

        self.param_env = outer_penv;
        self.tables = outer_tables;
        self.in_fn = outer_in_fn;
    }

    fn visit_pat(&mut self, p: &'tcx hir::Pat) {
        match p.node {
            PatKind::Lit(ref lit) => {
                self.check_const_eval(lit);
            }
            PatKind::Range(ref start, ref end) => {
                let const_cx = ConstContext::with_tables(self.tcx, self.tables);
                match const_cx.compare_lit_exprs(p.span, start, end) {
                    Ok(Ordering::Less) |
                    Ok(Ordering::Equal) => {}
                    Ok(Ordering::Greater) => {
                        struct_span_err!(self.tcx.sess, start.span, E0030,
                            "lower range bound must be less than or equal to upper")
                            .span_label(start.span, &format!("lower bound larger than upper bound"))
                            .emit();
                    }
                    Err(ErrorReported) => {}
                }
            }
            _ => {}
        }
        intravisit::walk_pat(self, p);
    }

    fn visit_stmt(&mut self, stmt: &'tcx hir::Stmt) {
        match stmt.node {
            hir::StmtDecl(ref decl, _) => {
                match decl.node {
                    hir::DeclLocal(_) => {
                        self.promotable = false;
                    }
                    // Item statements are allowed
                    hir::DeclItem(_) => {}
                }
            }
            hir::StmtExpr(..) |
            hir::StmtSemi(..) => {
                self.promotable = false;
            }
        }
        intravisit::walk_stmt(self, stmt);
    }

    fn visit_expr(&mut self, ex: &'tcx hir::Expr) {
        let outer = self.promotable;
        self.promotable = true;

        let node_ty = self.tables.node_id_to_type(ex.id);
        check_expr(self, ex, node_ty);
        check_adjustments(self, ex);

        if let hir::ExprMatch(ref discr, ref arms, _) = ex.node {
            // Compute the most demanding borrow from all the arms'
            // patterns and set that on the discriminator.
            let mut mut_borrow = false;
            for pat in arms.iter().flat_map(|arm| &arm.pats) {
                if self.mut_rvalue_borrows.remove(&pat.id) {
                    mut_borrow = true;
                }
            }
            if mut_borrow {
                self.mut_rvalue_borrows.insert(discr.id);
            }
        }

        intravisit::walk_expr(self, ex);

        // Handle borrows on (or inside the autorefs of) this expression.
        if self.mut_rvalue_borrows.remove(&ex.id) {
            self.promotable = false;
        }

        if self.in_fn && self.promotable {
            let const_cx = ConstContext::with_tables(self.tcx, self.tables);
            match const_cx.eval(ex, ExprTypeChecked) {
                Ok(_) => {}
                Err(ConstEvalErr { kind: UnimplementedConstVal(_), .. }) |
                Err(ConstEvalErr { kind: MiscCatchAll, .. }) |
                Err(ConstEvalErr { kind: MiscBinaryOp, .. }) |
                Err(ConstEvalErr { kind: NonConstPath, .. }) |
                Err(ConstEvalErr { kind: UnresolvedPath, .. }) |
                Err(ConstEvalErr { kind: ErroneousReferencedConstant(_), .. }) |
                Err(ConstEvalErr { kind: Math(ConstMathErr::Overflow(Op::Shr)), .. }) |
                Err(ConstEvalErr { kind: Math(ConstMathErr::Overflow(Op::Shl)), .. }) |
                Err(ConstEvalErr { kind: IndexOpFeatureGated, .. }) => {}
                Err(ConstEvalErr { kind: BadType(_), .. }) => {}
                Err(msg) => {
                    self.tcx.sess.add_lint(CONST_ERR,
                                           ex.id,
                                           msg.span,
                                           msg.description().into_oneline().into_owned())
                }
            }
        }

        self.tcx.rvalue_promotable_to_static.borrow_mut().insert(ex.id, self.promotable);
        self.promotable &= outer;
    }
}

/// This function is used to enforce the constraints on
/// const/static items. It walks through the *value*
/// of the item walking down the expression and evaluating
/// every nested expression. If the expression is not part
/// of a const/static item, it is qualified for promotion
/// instead of producing errors.
fn check_expr<'a, 'tcx>(v: &mut CheckCrateVisitor<'a, 'tcx>, e: &hir::Expr, node_ty: Ty<'tcx>) {
    match node_ty.sty {
        ty::TyAdt(def, _) if def.has_dtor() => {
            v.promotable = false;
        }
        _ => {}
    }

    let method_call = ty::MethodCall::expr(e.id);
    match e.node {
        hir::ExprUnary(..) |
        hir::ExprBinary(..) |
        hir::ExprIndex(..) if v.tables.method_map.contains_key(&method_call) => {
            v.promotable = false;
        }
        hir::ExprBox(_) => {
            v.promotable = false;
        }
        hir::ExprUnary(op, ref inner) => {
            match v.tables.node_id_to_type(inner.id).sty {
                ty::TyRawPtr(_) => {
                    assert!(op == hir::UnDeref);

                    v.promotable = false;
                }
                _ => {}
            }
        }
        hir::ExprBinary(op, ref lhs, _) => {
            match v.tables.node_id_to_type(lhs.id).sty {
                ty::TyRawPtr(_) => {
                    assert!(op.node == hir::BiEq || op.node == hir::BiNe ||
                            op.node == hir::BiLe || op.node == hir::BiLt ||
                            op.node == hir::BiGe || op.node == hir::BiGt);

                    v.promotable = false;
                }
                _ => {}
            }
        }
        hir::ExprCast(ref from, _) => {
            debug!("Checking const cast(id={})", from.id);
            match v.tcx.cast_kinds.borrow().get(&from.id) {
                None => span_bug!(e.span, "no kind for cast"),
                Some(&CastKind::PtrAddrCast) | Some(&CastKind::FnPtrAddrCast) => {
                    v.promotable = false;
                }
                _ => {}
            }
        }
        hir::ExprPath(ref qpath) => {
            let def = v.tables.qpath_def(qpath, e.id);
            match def {
                Def::VariantCtor(..) | Def::StructCtor(..) |
                Def::Fn(..) | Def::Method(..) => {}
                Def::AssociatedConst(_) => v.add_type(node_ty),
                Def::Const(did) => {
                    v.promotable &= if let Some(node_id) = v.tcx.map.as_local_node_id(did) {
                        match v.tcx.map.expect_item(node_id).node {
                            hir::ItemConst(_, body) => {
                                v.visit_nested_body(body);
                                v.tcx.rvalue_promotable_to_static.borrow()[&body.node_id]
                            }
                            _ => false
                        }
                    } else {
                        v.tcx.sess.cstore.const_is_rvalue_promotable_to_static(did)
                    };
                }
                _ => {
                    v.promotable = false;
                }
            }
        }
        hir::ExprCall(ref callee, _) => {
            let mut callee = &**callee;
            loop {
                callee = match callee.node {
                    hir::ExprBlock(ref block) => match block.expr {
                        Some(ref tail) => &tail,
                        None => break
                    },
                    _ => break
                };
            }
            // The callee is an arbitrary expression, it doesn't necessarily have a definition.
            let def = if let hir::ExprPath(ref qpath) = callee.node {
                v.tables.qpath_def(qpath, callee.id)
            } else {
                Def::Err
            };
            match def {
                Def::StructCtor(_, CtorKind::Fn) |
                Def::VariantCtor(_, CtorKind::Fn) => {}
                Def::Fn(did) => {
                    v.handle_const_fn_call(did, node_ty)
                }
                Def::Method(did) => {
                    match v.tcx.associated_item(did).container {
                        ty::ImplContainer(_) => {
                            v.handle_const_fn_call(did, node_ty)
                        }
                        ty::TraitContainer(_) => v.promotable = false
                    }
                }
                _ => v.promotable = false
            }
        }
        hir::ExprMethodCall(..) => {
            let method = v.tables.method_map[&method_call];
            match v.tcx.associated_item(method.def_id).container {
                ty::ImplContainer(_) => v.handle_const_fn_call(method.def_id, node_ty),
                ty::TraitContainer(_) => v.promotable = false
            }
        }
        hir::ExprStruct(..) => {
            if let ty::TyAdt(adt, ..) = v.tables.expr_ty(e).sty {
                // unsafe_cell_type doesn't necessarily exist with no_core
                if Some(adt.did) == v.tcx.lang_items.unsafe_cell_type() {
                    v.promotable = false;
                }
            }
        }

        hir::ExprLit(_) |
        hir::ExprAddrOf(..) |
        hir::ExprRepeat(..) => {}

        hir::ExprClosure(..) => {
            // Paths in constant contexts cannot refer to local variables,
            // as there are none, and thus closures can't have upvars there.
            if v.tcx.with_freevars(e.id, |fv| !fv.is_empty()) {
                v.promotable = false;
            }
        }

        hir::ExprBlock(_) |
        hir::ExprIndex(..) |
        hir::ExprField(..) |
        hir::ExprTupField(..) |
        hir::ExprArray(_) |
        hir::ExprType(..) |
        hir::ExprTup(..) => {}

        // Conditional control flow (possible to implement).
        hir::ExprMatch(..) |
        hir::ExprIf(..) |

        // Loops (not very meaningful in constants).
        hir::ExprWhile(..) |
        hir::ExprLoop(..) |

        // More control flow (also not very meaningful).
        hir::ExprBreak(..) |
        hir::ExprAgain(_) |
        hir::ExprRet(_) |

        // Expressions with side-effects.
        hir::ExprAssign(..) |
        hir::ExprAssignOp(..) |
        hir::ExprInlineAsm(..) => {
            v.promotable = false;
        }
    }
}

/// Check the adjustments of an expression
fn check_adjustments<'a, 'tcx>(v: &mut CheckCrateVisitor<'a, 'tcx>, e: &hir::Expr) {
    use rustc::ty::adjustment::*;

    match v.tables.adjustments.get(&e.id).map(|adj| adj.kind) {
        None |
        Some(Adjust::NeverToAny) |
        Some(Adjust::ReifyFnPointer) |
        Some(Adjust::UnsafeFnPointer) |
        Some(Adjust::MutToConstPointer) => {}

        Some(Adjust::DerefRef { autoderefs, .. }) => {
            if (0..autoderefs as u32)
                .any(|autoderef| v.tables.is_overloaded_autoderef(e.id, autoderef)) {
                v.promotable = false;
            }
        }
    }
}

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    tcx.visit_all_item_likes_in_krate(DepNode::CheckConst,
                                      &mut CheckCrateVisitor {
                                          tcx: tcx,
                                          tables: &ty::Tables::empty(),
                                          in_fn: false,
                                          promotable: false,
                                          mut_rvalue_borrows: NodeSet(),
                                          param_env: tcx.empty_parameter_environment(),
                                      }.as_deep_visitor());
    tcx.sess.abort_if_errors();
}

impl<'a, 'gcx, 'tcx> euv::Delegate<'tcx> for CheckCrateVisitor<'a, 'gcx> {
    fn consume(&mut self,
               _consume_id: ast::NodeId,
               _consume_span: Span,
               _cmt: mc::cmt,
               _mode: euv::ConsumeMode) {}

    fn borrow(&mut self,
              borrow_id: ast::NodeId,
              _borrow_span: Span,
              cmt: mc::cmt<'tcx>,
              _loan_region: &'tcx ty::Region,
              bk: ty::BorrowKind,
              loan_cause: euv::LoanCause) {
        // Kind of hacky, but we allow Unsafe coercions in constants.
        // These occur when we convert a &T or *T to a *U, as well as
        // when making a thin pointer (e.g., `*T`) into a fat pointer
        // (e.g., `*Trait`).
        match loan_cause {
            euv::LoanCause::AutoUnsafe => {
                return;
            }
            _ => {}
        }

        let mut cur = &cmt;
        loop {
            match cur.cat {
                Categorization::Rvalue(..) => {
                    if loan_cause == euv::MatchDiscriminant {
                        // Ignore the dummy immutable borrow created by EUV.
                        break;
                    }
                    if bk.to_mutbl_lossy() == hir::MutMutable {
                        self.mut_rvalue_borrows.insert(borrow_id);
                    }
                    break;
                }
                Categorization::StaticItem => {
                    break;
                }
                Categorization::Deref(ref cmt, ..) |
                Categorization::Downcast(ref cmt, _) |
                Categorization::Interior(ref cmt, _) => {
                    cur = cmt;
                }

                Categorization::Upvar(..) |
                Categorization::Local(..) => break,
            }
        }
    }

    fn decl_without_init(&mut self, _id: ast::NodeId, _span: Span) {}
    fn mutate(&mut self,
              _assignment_id: ast::NodeId,
              _assignment_span: Span,
              _assignee_cmt: mc::cmt,
              _mode: euv::MutateMode) {
    }

    fn matched_pat(&mut self, _: &hir::Pat, _: mc::cmt, _: euv::MatchMode) {}

    fn consume_pat(&mut self, _consume_pat: &hir::Pat, _cmt: mc::cmt, _mode: euv::ConsumeMode) {}
}
