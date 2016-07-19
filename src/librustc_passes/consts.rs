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
use rustc::ty::cast::{CastKind};
use rustc_const_eval::{ConstEvalErr, lookup_const_fn_by_id, compare_lit_exprs};
use rustc_const_eval::{eval_const_expr_partial, lookup_const_by_id};
use rustc_const_eval::ErrKind::{IndexOpFeatureGated, UnimplementedConstVal, MiscCatchAll, Math};
use rustc_const_eval::ErrKind::{ErroneousReferencedConstant, MiscBinaryOp, NonConstPath};
use rustc_const_eval::ErrKind::UnresolvedPath;
use rustc_const_eval::EvalHint::ExprTypeChecked;
use rustc_const_math::{ConstMathErr, Op};
use rustc::hir::def::Def;
use rustc::hir::def_id::DefId;
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use rustc::middle::mem_categorization::Categorization;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::traits::ProjectionMode;
use rustc::util::nodemap::NodeMap;
use rustc::middle::const_qualif::ConstQualif;
use rustc::lint::builtin::CONST_ERR;

use rustc::hir::{self, PatKind};
use syntax::ast;
use syntax_pos::Span;
use rustc::hir::intravisit::{self, FnKind, Visitor};

use std::collections::hash_map::Entry;
use std::cmp::Ordering;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Mode {
    Const,
    ConstFn,
    Static,
    StaticMut,

    // An expression that occurs outside of any constant context
    // (i.e. `const`, `static`, array lengths, etc.). The value
    // can be variable at runtime, but will be promotable to
    // static memory if we can prove it is actually constant.
    Var,
}

struct CheckCrateVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mode: Mode,
    qualif: ConstQualif,
    rvalue_borrows: NodeMap<hir::Mutability>
}

impl<'a, 'gcx> CheckCrateVisitor<'a, 'gcx> {
    fn with_mode<F, R>(&mut self, mode: Mode, f: F) -> R where
        F: FnOnce(&mut CheckCrateVisitor<'a, 'gcx>) -> R,
    {
        let (old_mode, old_qualif) = (self.mode, self.qualif);
        self.mode = mode;
        self.qualif = ConstQualif::empty();
        let r = f(self);
        self.mode = old_mode;
        self.qualif = old_qualif;
        r
    }

    fn with_euv<F, R>(&mut self, item_id: Option<ast::NodeId>, f: F) -> R where
        F: for<'b, 'tcx> FnOnce(&mut euv::ExprUseVisitor<'b, 'gcx, 'tcx>) -> R,
    {
        let param_env = match item_id {
            Some(item_id) => ty::ParameterEnvironment::for_item(self.tcx, item_id),
            None => self.tcx.empty_parameter_environment()
        };

        self.tcx.infer_ctxt(None, Some(param_env), ProjectionMode::AnyFinal).enter(|infcx| {
            f(&mut euv::ExprUseVisitor::new(self, &infcx))
        })
    }

    fn global_expr(&mut self, mode: Mode, expr: &hir::Expr) -> ConstQualif {
        assert!(mode != Mode::Var);
        match self.tcx.const_qualif_map.borrow_mut().entry(expr.id) {
            Entry::Occupied(entry) => return *entry.get(),
            Entry::Vacant(entry) => {
                // Prevent infinite recursion on re-entry.
                entry.insert(ConstQualif::empty());
            }
        }
        if let Err(err) = eval_const_expr_partial(self.tcx, expr, ExprTypeChecked, None) {
            match err.kind {
                UnimplementedConstVal(_) => {},
                IndexOpFeatureGated => {},
                ErroneousReferencedConstant(_) => {},
                _ => self.tcx.sess.add_lint(CONST_ERR, expr.id, expr.span,
                                         format!("constant evaluation error: {}. This will \
                                                 become a HARD ERROR in the future",
                                                 err.description())),
            }
        }
        self.with_mode(mode, |this| {
            this.with_euv(None, |euv| euv.consume_expr(expr));
            this.visit_expr(expr);
            this.qualif
        })
    }

    fn fn_like(&mut self,
               fk: FnKind,
               fd: &hir::FnDecl,
               b: &hir::Block,
               s: Span,
               fn_id: ast::NodeId)
               -> ConstQualif {
        match self.tcx.const_qualif_map.borrow_mut().entry(fn_id) {
            Entry::Occupied(entry) => return *entry.get(),
            Entry::Vacant(entry) => {
                // Prevent infinite recursion on re-entry.
                entry.insert(ConstQualif::empty());
            }
        }

        let mode = match fk {
            FnKind::ItemFn(_, _, _, hir::Constness::Const, _, _, _) => {
                Mode::ConstFn
            }
            FnKind::Method(_, m, _, _) => {
                if m.constness == hir::Constness::Const {
                    Mode::ConstFn
                } else {
                    Mode::Var
                }
            }
            _ => Mode::Var
        };

        let qualif = self.with_mode(mode, |this| {
            this.with_euv(Some(fn_id), |euv| euv.walk_fn(fd, b));
            intravisit::walk_fn(this, fk, fd, b, s);
            this.qualif
        });

        // Keep only bits that aren't affected by function body (NON_ZERO_SIZED),
        // and bits that don't change semantics, just optimizations (PREFER_IN_PLACE).
        let qualif = qualif & (ConstQualif::NON_ZERO_SIZED | ConstQualif::PREFER_IN_PLACE);

        self.tcx.const_qualif_map.borrow_mut().insert(fn_id, qualif);
        qualif
    }

    fn add_qualif(&mut self, qualif: ConstQualif) {
        self.qualif = self.qualif | qualif;
    }

    /// Returns true if the call is to a const fn or method.
    fn handle_const_fn_call(&mut self,
                            _expr: &hir::Expr,
                            def_id: DefId,
                            ret_ty: Ty<'gcx>)
                            -> bool {
        if let Some(fn_like) = lookup_const_fn_by_id(self.tcx, def_id) {
            let qualif = self.fn_like(fn_like.kind(),
                                      fn_like.decl(),
                                      fn_like.body(),
                                      fn_like.span(),
                                      fn_like.id());
            self.add_qualif(qualif);

            if ret_ty.type_contents(self.tcx).interior_unsafe() {
                self.add_qualif(ConstQualif::MUTABLE_MEM);
            }

            true
        } else {
            false
        }
    }

    fn record_borrow(&mut self, id: ast::NodeId, mutbl: hir::Mutability) {
        match self.rvalue_borrows.entry(id) {
            Entry::Occupied(mut entry) => {
                // Merge the two borrows, taking the most demanding
                // one, mutability-wise.
                if mutbl == hir::MutMutable {
                    entry.insert(mutbl);
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(mutbl);
            }
        }
    }

    fn msg(&self) -> &'static str {
        match self.mode {
            Mode::Const => "constant",
            Mode::ConstFn => "constant function",
            Mode::StaticMut | Mode::Static => "static",
            Mode::Var => bug!(),
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for CheckCrateVisitor<'a, 'tcx> {
    fn visit_item(&mut self, i: &hir::Item) {
        debug!("visit_item(item={})", self.tcx.map.node_to_string(i.id));
        assert_eq!(self.mode, Mode::Var);
        match i.node {
            hir::ItemStatic(_, hir::MutImmutable, ref expr) => {
                self.global_expr(Mode::Static, &expr);
            }
            hir::ItemStatic(_, hir::MutMutable, ref expr) => {
                self.global_expr(Mode::StaticMut, &expr);
            }
            hir::ItemConst(_, ref expr) => {
                self.global_expr(Mode::Const, &expr);
            }
            hir::ItemEnum(ref enum_definition, _) => {
                for var in &enum_definition.variants {
                    if let Some(ref ex) = var.node.disr_expr {
                        self.global_expr(Mode::Const, &ex);
                    }
                }
            }
            _ => {
                intravisit::walk_item(self, i);
            }
        }
    }

    fn visit_trait_item(&mut self, t: &'v hir::TraitItem) {
        match t.node {
            hir::ConstTraitItem(_, ref default) => {
                if let Some(ref expr) = *default {
                    self.global_expr(Mode::Const, &expr);
                } else {
                    intravisit::walk_trait_item(self, t);
                }
            }
            _ => self.with_mode(Mode::Var, |v| intravisit::walk_trait_item(v, t)),
        }
    }

    fn visit_impl_item(&mut self, i: &'v hir::ImplItem) {
        match i.node {
            hir::ImplItemKind::Const(_, ref expr) => {
                self.global_expr(Mode::Const, &expr);
            }
            _ => self.with_mode(Mode::Var, |v| intravisit::walk_impl_item(v, i)),
        }
    }

    fn visit_fn(&mut self,
                fk: FnKind<'v>,
                fd: &'v hir::FnDecl,
                b: &'v hir::Block,
                s: Span,
                fn_id: ast::NodeId) {
        self.fn_like(fk, fd, b, s, fn_id);
    }

    fn visit_pat(&mut self, p: &hir::Pat) {
        match p.node {
            PatKind::Lit(ref lit) => {
                self.global_expr(Mode::Const, &lit);
            }
            PatKind::Range(ref start, ref end) => {
                self.global_expr(Mode::Const, &start);
                self.global_expr(Mode::Const, &end);

                match compare_lit_exprs(self.tcx, start, end) {
                    Some(Ordering::Less) |
                    Some(Ordering::Equal) => {}
                    Some(Ordering::Greater) => {
                        span_err!(self.tcx.sess, start.span, E0030,
                            "lower range bound must be less than or equal to upper");
                    }
                    None => {
                        span_err!(self.tcx.sess, p.span, E0014,
                                  "paths in {}s may only refer to constants",
                                  self.msg());
                    }
                }
            }
            _ => intravisit::walk_pat(self, p)
        }
    }

    fn visit_block(&mut self, block: &hir::Block) {
        // Check all statements in the block
        for stmt in &block.stmts {
            match stmt.node {
                hir::StmtDecl(ref decl, _) => {
                    match decl.node {
                        hir::DeclLocal(_) => {},
                        // Item statements are allowed
                        hir::DeclItem(_) => continue
                    }
                }
                hir::StmtExpr(_, _) => {},
                hir::StmtSemi(_, _) => {},
            }
            self.add_qualif(ConstQualif::NOT_CONST);
        }
        intravisit::walk_block(self, block);
    }

    fn visit_expr(&mut self, ex: &hir::Expr) {
        let mut outer = self.qualif;
        self.qualif = ConstQualif::empty();

        let node_ty = self.tcx.node_id_to_type(ex.id);
        check_expr(self, ex, node_ty);
        check_adjustments(self, ex);

        // Special-case some expressions to avoid certain flags bubbling up.
        match ex.node {
            hir::ExprCall(ref callee, ref args) => {
                for arg in args {
                    self.visit_expr(&arg)
                }

                let inner = self.qualif;
                self.visit_expr(&callee);
                // The callee's size doesn't count in the call.
                let added = self.qualif - inner;
                self.qualif = inner | (added - ConstQualif::NON_ZERO_SIZED);
            }
            hir::ExprRepeat(ref element, _) => {
                self.visit_expr(&element);
                // The count is checked elsewhere (typeck).
                let count = match node_ty.sty {
                    ty::TyArray(_, n) => n,
                    _ => bug!()
                };
                // [element; 0] is always zero-sized.
                if count == 0 {
                    self.qualif.remove(ConstQualif::NON_ZERO_SIZED | ConstQualif::PREFER_IN_PLACE);
                }
            }
            hir::ExprMatch(ref discr, ref arms, _) => {
                // Compute the most demanding borrow from all the arms'
                // patterns and set that on the discriminator.
                let mut borrow = None;
                for pat in arms.iter().flat_map(|arm| &arm.pats) {
                    let pat_borrow = self.rvalue_borrows.remove(&pat.id);
                    match (borrow, pat_borrow) {
                        (None, _) | (_, Some(hir::MutMutable)) => {
                            borrow = pat_borrow;
                        }
                        _ => {}
                    }
                }
                if let Some(mutbl) = borrow {
                    self.record_borrow(discr.id, mutbl);
                }
                intravisit::walk_expr(self, ex);
            }
            _ => intravisit::walk_expr(self, ex)
        }

        // Handle borrows on (or inside the autorefs of) this expression.
        match self.rvalue_borrows.remove(&ex.id) {
            Some(hir::MutImmutable) => {
                // Constants cannot be borrowed if they contain interior mutability as
                // it means that our "silent insertion of statics" could change
                // initializer values (very bad).
                // If the type doesn't have interior mutability, then `ConstQualif::MUTABLE_MEM` has
                // propagated from another error, so erroring again would be just noise.
                let tc = node_ty.type_contents(self.tcx);
                if self.qualif.intersects(ConstQualif::MUTABLE_MEM) && tc.interior_unsafe() {
                    outer = outer | ConstQualif::NOT_CONST;
                }
                // If the reference has to be 'static, avoid in-place initialization
                // as that will end up pointing to the stack instead.
                if !self.qualif.intersects(ConstQualif::NON_STATIC_BORROWS) {
                    self.qualif = self.qualif - ConstQualif::PREFER_IN_PLACE;
                    self.add_qualif(ConstQualif::HAS_STATIC_BORROWS);
                }
            }
            Some(hir::MutMutable) => {
                // `&mut expr` means expr could be mutated, unless it's zero-sized.
                if self.qualif.intersects(ConstQualif::NON_ZERO_SIZED) {
                    if self.mode == Mode::Var {
                        outer = outer | ConstQualif::NOT_CONST;
                        self.add_qualif(ConstQualif::MUTABLE_MEM);
                    }
                }
                if !self.qualif.intersects(ConstQualif::NON_STATIC_BORROWS) {
                    self.add_qualif(ConstQualif::HAS_STATIC_BORROWS);
                }
            }
            None => {}
        }

        if self.mode == Mode::Var && !self.qualif.intersects(ConstQualif::NOT_CONST) {
            match eval_const_expr_partial(self.tcx, ex, ExprTypeChecked, None) {
                Ok(_) => {}
                Err(ConstEvalErr { kind: UnimplementedConstVal(_), ..}) |
                Err(ConstEvalErr { kind: MiscCatchAll, ..}) |
                Err(ConstEvalErr { kind: MiscBinaryOp, ..}) |
                Err(ConstEvalErr { kind: NonConstPath, ..}) |
                Err(ConstEvalErr { kind: UnresolvedPath, ..}) |
                Err(ConstEvalErr { kind: ErroneousReferencedConstant(_), ..}) |
                Err(ConstEvalErr { kind: Math(ConstMathErr::Overflow(Op::Shr)), ..}) |
                Err(ConstEvalErr { kind: Math(ConstMathErr::Overflow(Op::Shl)), ..}) |
                Err(ConstEvalErr { kind: IndexOpFeatureGated, ..}) => {},
                Err(msg) => {
                    self.tcx.sess.add_lint(CONST_ERR, ex.id,
                                           msg.span,
                                           msg.description().into_owned())
                }
            }
        }

        self.tcx.const_qualif_map.borrow_mut().insert(ex.id, self.qualif);
        // Don't propagate certain flags.
        self.qualif = outer | (self.qualif - ConstQualif::HAS_STATIC_BORROWS);
    }
}

/// This function is used to enforce the constraints on
/// const/static items. It walks through the *value*
/// of the item walking down the expression and evaluating
/// every nested expression. If the expression is not part
/// of a const/static item, it is qualified for promotion
/// instead of producing errors.
fn check_expr<'a, 'tcx>(v: &mut CheckCrateVisitor<'a, 'tcx>,
                        e: &hir::Expr, node_ty: Ty<'tcx>) {
    match node_ty.sty {
        ty::TyStruct(def, _) |
        ty::TyEnum(def, _) if def.has_dtor() => {
            v.add_qualif(ConstQualif::NEEDS_DROP);
        }
        _ => {}
    }

    let method_call = ty::MethodCall::expr(e.id);
    match e.node {
        hir::ExprUnary(..) |
        hir::ExprBinary(..) |
        hir::ExprIndex(..) if v.tcx.tables.borrow().method_map.contains_key(&method_call) => {
            v.add_qualif(ConstQualif::NOT_CONST);
        }
        hir::ExprBox(_) => {
            v.add_qualif(ConstQualif::NOT_CONST);
        }
        hir::ExprUnary(op, ref inner) => {
            match v.tcx.node_id_to_type(inner.id).sty {
                ty::TyRawPtr(_) => {
                    assert!(op == hir::UnDeref);

                    v.add_qualif(ConstQualif::NOT_CONST);
                }
                _ => {}
            }
        }
        hir::ExprBinary(op, ref lhs, _) => {
            match v.tcx.node_id_to_type(lhs.id).sty {
                ty::TyRawPtr(_) => {
                    assert!(op.node == hir::BiEq || op.node == hir::BiNe ||
                            op.node == hir::BiLe || op.node == hir::BiLt ||
                            op.node == hir::BiGe || op.node == hir::BiGt);

                    v.add_qualif(ConstQualif::NOT_CONST);
                }
                _ => {}
            }
        }
        hir::ExprCast(ref from, _) => {
            debug!("Checking const cast(id={})", from.id);
            match v.tcx.cast_kinds.borrow().get(&from.id) {
                None => span_bug!(e.span, "no kind for cast"),
                Some(&CastKind::PtrAddrCast) | Some(&CastKind::FnPtrAddrCast) => {
                    v.add_qualif(ConstQualif::NOT_CONST);
                }
                _ => {}
            }
        }
        hir::ExprPath(..) => {
            match v.tcx.expect_def(e.id) {
                Def::Variant(..) => {
                    // Count the discriminator or function pointer.
                    v.add_qualif(ConstQualif::NON_ZERO_SIZED);
                }
                Def::Struct(..) => {
                    if let ty::TyFnDef(..) = node_ty.sty {
                        // Count the function pointer.
                        v.add_qualif(ConstQualif::NON_ZERO_SIZED);
                    }
                }
                Def::Fn(..) | Def::Method(..) => {
                    // Count the function pointer.
                    v.add_qualif(ConstQualif::NON_ZERO_SIZED);
                }
                Def::Static(..) => {
                    match v.mode {
                        Mode::Static | Mode::StaticMut => {}
                        Mode::Const | Mode::ConstFn => {}
                        Mode::Var => v.add_qualif(ConstQualif::NOT_CONST)
                    }
                }
                Def::Const(did) | Def::AssociatedConst(did) => {
                    let substs = Some(v.tcx.node_id_item_substs(e.id).substs);
                    if let Some((expr, _)) = lookup_const_by_id(v.tcx, did, substs) {
                        let inner = v.global_expr(Mode::Const, expr);
                        v.add_qualif(inner);
                    }
                }
                Def::Local(..) if v.mode == Mode::ConstFn => {
                    // Sadly, we can't determine whether the types are zero-sized.
                    v.add_qualif(ConstQualif::NOT_CONST | ConstQualif::NON_ZERO_SIZED);
                }
                _ => {
                    v.add_qualif(ConstQualif::NOT_CONST);
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
            let is_const = match v.tcx.expect_def_or_none(callee.id) {
                Some(Def::Struct(..)) => true,
                Some(Def::Variant(..)) => {
                    // Count the discriminator.
                    v.add_qualif(ConstQualif::NON_ZERO_SIZED);
                    true
                }
                Some(Def::Fn(did)) => {
                    v.handle_const_fn_call(e, did, node_ty)
                }
                Some(Def::Method(did)) => {
                    match v.tcx.impl_or_trait_item(did).container() {
                        ty::ImplContainer(_) => {
                            v.handle_const_fn_call(e, did, node_ty)
                        }
                        ty::TraitContainer(_) => false
                    }
                }
                _ => false
            };
            if !is_const {
                v.add_qualif(ConstQualif::NOT_CONST);
            }
        }
        hir::ExprMethodCall(..) => {
            let method = v.tcx.tables.borrow().method_map[&method_call];
            let is_const = match v.tcx.impl_or_trait_item(method.def_id).container() {
                ty::ImplContainer(_) => v.handle_const_fn_call(e, method.def_id, node_ty),
                ty::TraitContainer(_) => false
            };
            if !is_const {
                v.add_qualif(ConstQualif::NOT_CONST);
            }
        }
        hir::ExprStruct(..) => {
            // unsafe_cell_type doesn't necessarily exist with no_core
            if Some(v.tcx.expect_def(e.id).def_id()) == v.tcx.lang_items.unsafe_cell_type() {
                v.add_qualif(ConstQualif::MUTABLE_MEM);
            }
        }

        hir::ExprLit(_) |
        hir::ExprAddrOf(..) => {
            v.add_qualif(ConstQualif::NON_ZERO_SIZED);
        }

        hir::ExprRepeat(..) => {
            v.add_qualif(ConstQualif::PREFER_IN_PLACE);
        }

        hir::ExprClosure(..) => {
            // Paths in constant contexts cannot refer to local variables,
            // as there are none, and thus closures can't have upvars there.
            if v.tcx.with_freevars(e.id, |fv| !fv.is_empty()) {
                assert!(v.mode == Mode::Var,
                        "global closures can't capture anything");
                v.add_qualif(ConstQualif::NOT_CONST);
            }
        }

        hir::ExprBlock(_) |
        hir::ExprIndex(..) |
        hir::ExprField(..) |
        hir::ExprTupField(..) |
        hir::ExprVec(_) |
        hir::ExprType(..) |
        hir::ExprTup(..) => {}

        // Conditional control flow (possible to implement).
        hir::ExprMatch(..) |
        hir::ExprIf(..) |

        // Loops (not very meaningful in constants).
        hir::ExprWhile(..) |
        hir::ExprLoop(..) |

        // More control flow (also not very meaningful).
        hir::ExprBreak(_) |
        hir::ExprAgain(_) |
        hir::ExprRet(_) |

        // Expressions with side-effects.
        hir::ExprAssign(..) |
        hir::ExprAssignOp(..) |
        hir::ExprInlineAsm(..) => {
            v.add_qualif(ConstQualif::NOT_CONST);
        }
    }
}

/// Check the adjustments of an expression
fn check_adjustments<'a, 'tcx>(v: &mut CheckCrateVisitor<'a, 'tcx>, e: &hir::Expr) {
    match v.tcx.tables.borrow().adjustments.get(&e.id) {
        None |
        Some(&ty::adjustment::AdjustReifyFnPointer) |
        Some(&ty::adjustment::AdjustUnsafeFnPointer) |
        Some(&ty::adjustment::AdjustMutToConstPointer) => {}

        Some(&ty::adjustment::AdjustDerefRef(
            ty::adjustment::AutoDerefRef { autoderefs, .. }
        )) => {
            if (0..autoderefs as u32).any(|autoderef| {
                    v.tcx.is_overloaded_autoderef(e.id, autoderef)
            }) {
                v.add_qualif(ConstQualif::NOT_CONST);
            }
        }
    }
}

pub fn check_crate<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>) {
    tcx.visit_all_items_in_krate(DepNode::CheckConst, &mut CheckCrateVisitor {
        tcx: tcx,
        mode: Mode::Var,
        qualif: ConstQualif::NOT_CONST,
        rvalue_borrows: NodeMap()
    });
    tcx.sess.abort_if_errors();
}

impl<'a, 'gcx, 'tcx> euv::Delegate<'tcx> for CheckCrateVisitor<'a, 'gcx> {
    fn consume(&mut self,
               _consume_id: ast::NodeId,
               _consume_span: Span,
               cmt: mc::cmt,
               _mode: euv::ConsumeMode) {
        let mut cur = &cmt;
        loop {
            match cur.cat {
                Categorization::StaticItem => {
                    break;
                }
                Categorization::Deref(ref cmt, _, _) |
                Categorization::Downcast(ref cmt, _) |
                Categorization::Interior(ref cmt, _) => cur = cmt,

                Categorization::Rvalue(..) |
                Categorization::Upvar(..) |
                Categorization::Local(..) => break
            }
        }
    }
    fn borrow(&mut self,
              borrow_id: ast::NodeId,
              _borrow_span: Span,
              cmt: mc::cmt<'tcx>,
              _loan_region: ty::Region,
              bk: ty::BorrowKind,
              loan_cause: euv::LoanCause)
    {
        // Kind of hacky, but we allow Unsafe coercions in constants.
        // These occur when we convert a &T or *T to a *U, as well as
        // when making a thin pointer (e.g., `*T`) into a fat pointer
        // (e.g., `*Trait`).
        match loan_cause {
            euv::LoanCause::AutoUnsafe => {
                return;
            }
            _ => { }
        }

        let mut cur = &cmt;
        loop {
            match cur.cat {
                Categorization::Rvalue(..) => {
                    if loan_cause == euv::MatchDiscriminant {
                        // Ignore the dummy immutable borrow created by EUV.
                        break;
                    }
                    let mutbl = bk.to_mutbl_lossy();
                    if mutbl == hir::MutMutable && self.mode == Mode::StaticMut {
                        // Mutable slices are the only `&mut` allowed in
                        // globals, but only in `static mut`, nowhere else.
                        // FIXME: This exception is really weird... there isn't
                        // any fundamental reason to restrict this based on
                        // type of the expression.  `&mut [1]` has exactly the
                        // same representation as &mut 1.
                        match cmt.ty.sty {
                            ty::TyArray(_, _) | ty::TySlice(_) => break,
                            _ => {}
                        }
                    }
                    self.record_borrow(borrow_id, mutbl);
                    break;
                }
                Categorization::StaticItem => {
                    break;
                }
                Categorization::Deref(ref cmt, _, _) |
                Categorization::Downcast(ref cmt, _) |
                Categorization::Interior(ref cmt, _) => {
                    cur = cmt;
                }

                Categorization::Upvar(..) |
                Categorization::Local(..) => break
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
                   _: &hir::Pat,
                   _: mc::cmt,
                   _: euv::MatchMode) {}

    fn consume_pat(&mut self,
                   _consume_pat: &hir::Pat,
                   _cmt: mc::cmt,
                   _mode: euv::ConsumeMode) {}
}
