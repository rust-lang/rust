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

use dep_graph::DepNode;
use middle::ty::cast::{CastKind};
use middle::const_eval::{self, ConstEvalErr};
use middle::const_eval::ErrKind::IndexOpFeatureGated;
use middle::const_eval::EvalHint::ExprTypeChecked;
use middle::def;
use middle::def_id::DefId;
use middle::expr_use_visitor as euv;
use middle::infer;
use middle::mem_categorization as mc;
use middle::mem_categorization::Categorization;
use middle::traits;
use middle::ty::{self, Ty};
use util::nodemap::NodeMap;

use rustc_front::hir;
use syntax::ast;
use syntax::codemap::Span;
use syntax::feature_gate::UnstableFeatures;
use rustc_front::intravisit::{self, FnKind, Visitor};

use std::collections::hash_map::Entry;
use std::cmp::Ordering;

// Const qualification, from partial to completely promotable.
bitflags! {
    #[derive(RustcEncodable, RustcDecodable)]
    flags ConstQualif: u8 {
        // Inner mutability (can not be placed behind a reference) or behind
        // &mut in a non-global expression. Can be copied from static memory.
        const MUTABLE_MEM        = 1 << 0,
        // Constant value with a type that implements Drop. Can be copied
        // from static memory, similar to MUTABLE_MEM.
        const NEEDS_DROP         = 1 << 1,
        // Even if the value can be placed in static memory, copying it from
        // there is more expensive than in-place instantiation, and/or it may
        // be too large. This applies to [T; N] and everything containing it.
        // N.B.: references need to clear this flag to not end up on the stack.
        const PREFER_IN_PLACE    = 1 << 2,
        // May use more than 0 bytes of memory, doesn't impact the constness
        // directly, but is not allowed to be borrowed mutably in a constant.
        const NON_ZERO_SIZED     = 1 << 3,
        // Actually borrowed, has to always be in static memory. Does not
        // propagate, and requires the expression to behave like a 'static
        // lvalue. The set of expressions with this flag is the minimum
        // that have to be promoted.
        const HAS_STATIC_BORROWS = 1 << 4,
        // Invalid const for miscellaneous reasons (e.g. not implemented).
        const NOT_CONST          = 1 << 5,

        // Borrowing the expression won't produce &'static T if any of these
        // bits are set, though the value could be copied from static memory
        // if `NOT_CONST` isn't set.
        const NON_STATIC_BORROWS = ConstQualif::MUTABLE_MEM.bits |
                                   ConstQualif::NEEDS_DROP.bits |
                                   ConstQualif::NOT_CONST.bits
    }
}

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
    tcx: &'a ty::ctxt<'tcx>,
    mode: Mode,
    qualif: ConstQualif,
    rvalue_borrows: NodeMap<hir::Mutability>
}

impl<'a, 'tcx> CheckCrateVisitor<'a, 'tcx> {
    fn with_mode<F, R>(&mut self, mode: Mode, f: F) -> R where
        F: FnOnce(&mut CheckCrateVisitor<'a, 'tcx>) -> R,
    {
        let (old_mode, old_qualif) = (self.mode, self.qualif);
        self.mode = mode;
        self.qualif = ConstQualif::empty();
        let r = f(self);
        self.mode = old_mode;
        self.qualif = old_qualif;
        r
    }

    fn with_euv<'b, F, R>(&'b mut self, item_id: Option<ast::NodeId>, f: F) -> R where
        F: for<'t> FnOnce(&mut euv::ExprUseVisitor<'b, 't, 'b, 'tcx>) -> R,
    {
        let param_env = match item_id {
            Some(item_id) => ty::ParameterEnvironment::for_item(self.tcx, item_id),
            None => self.tcx.empty_parameter_environment()
        };

        let infcx = infer::new_infer_ctxt(self.tcx, &self.tcx.tables, Some(param_env));

        f(&mut euv::ExprUseVisitor::new(self, &infcx))
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
            FnKind::ItemFn(_, _, _, hir::Constness::Const, _, _) => {
                Mode::ConstFn
            }
            FnKind::Method(_, m, _) => {
                if m.constness == hir::Constness::Const {
                    Mode::ConstFn
                } else {
                    Mode::Var
                }
            }
            _ => Mode::Var
        };

        // Ensure the arguments are simple, not mutable/by-ref or patterns.
        if mode == Mode::ConstFn {
            for arg in &fd.inputs {
                match arg.pat.node {
                    hir::PatWild => {}
                    hir::PatIdent(hir::BindByValue(hir::MutImmutable), _, None) => {}
                    _ => {
                        span_err!(self.tcx.sess, arg.pat.span, E0022,
                                  "arguments of constant functions can only \
                                   be immutable by-value bindings");
                    }
                }
            }
        }

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
                            expr: &hir::Expr,
                            def_id: DefId,
                            ret_ty: Ty<'tcx>)
                            -> bool {
        if let Some(fn_like) = const_eval::lookup_const_fn_by_id(self.tcx, def_id) {
            if
                // we are in a static/const initializer
                self.mode != Mode::Var &&

                // feature-gate is not enabled
                !self.tcx.sess.features.borrow().const_fn &&

                // this doesn't come from a macro that has #[allow_internal_unstable]
                !self.tcx.sess.codemap().span_allows_unstable(expr.span)
            {
                let mut err = self.tcx.sess.struct_span_err(
                    expr.span,
                    "const fns are an unstable feature");
                fileline_help!(
                    &mut err,
                    expr.span,
                    "in Nightly builds, add `#![feature(const_fn)]` to the crate \
                     attributes to enable");
                err.emit();
            }

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
            Mode::Var => unreachable!(),
        }
    }

    fn check_static_mut_type(&self, e: &hir::Expr) {
        let node_ty = self.tcx.node_id_to_type(e.id);
        let tcontents = node_ty.type_contents(self.tcx);

        let suffix = if tcontents.has_dtor() {
            "destructors"
        } else if tcontents.owns_owned() {
            "boxes"
        } else {
            return
        };

        span_err!(self.tcx.sess, e.span, E0397,
                 "mutable statics are not allowed to have {}", suffix);
    }

    fn check_static_type(&self, e: &hir::Expr) {
        let ty = self.tcx.node_id_to_type(e.id);
        let infcx = infer::new_infer_ctxt(self.tcx, &self.tcx.tables, None);
        let cause = traits::ObligationCause::new(e.span, e.id, traits::SharedStatic);
        let mut fulfill_cx = infcx.fulfillment_cx.borrow_mut();
        fulfill_cx.register_builtin_bound(&infcx, ty, ty::BoundSync, cause);
        match fulfill_cx.select_all_or_error(&infcx) {
            Ok(()) => { },
            Err(ref errors) => {
                traits::report_fulfillment_errors(&infcx, errors);
            }
        }
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for CheckCrateVisitor<'a, 'tcx> {
    fn visit_item(&mut self, i: &hir::Item) {
        debug!("visit_item(item={})", self.tcx.map.node_to_string(i.id));
        assert_eq!(self.mode, Mode::Var);
        match i.node {
            hir::ItemStatic(_, hir::MutImmutable, ref expr) => {
                self.check_static_type(&**expr);
                self.global_expr(Mode::Static, &**expr);
            }
            hir::ItemStatic(_, hir::MutMutable, ref expr) => {
                self.check_static_mut_type(&**expr);
                self.global_expr(Mode::StaticMut, &**expr);
            }
            hir::ItemConst(_, ref expr) => {
                self.global_expr(Mode::Const, &**expr);
            }
            hir::ItemEnum(ref enum_definition, _) => {
                for var in &enum_definition.variants {
                    if let Some(ref ex) = var.node.disr_expr {
                        self.global_expr(Mode::Const, &**ex);
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
                    self.global_expr(Mode::Const, &*expr);
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
                self.global_expr(Mode::Const, &*expr);
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
            hir::PatLit(ref lit) => {
                self.global_expr(Mode::Const, &**lit);
            }
            hir::PatRange(ref start, ref end) => {
                self.global_expr(Mode::Const, &**start);
                self.global_expr(Mode::Const, &**end);

                match const_eval::compare_lit_exprs(self.tcx, start, end) {
                    Some(Ordering::Less) |
                    Some(Ordering::Equal) => {}
                    Some(Ordering::Greater) => {
                        span_err!(self.tcx.sess, start.span, E0030,
                            "lower range bound must be less than or equal to upper");
                    }
                    None => {
                        self.tcx.sess.delay_span_bug(start.span,
                                                     "non-constant path in constant expr");
                    }
                }
            }
            _ => intravisit::walk_pat(self, p)
        }
    }

    fn visit_block(&mut self, block: &hir::Block) {
        // Check all statements in the block
        for stmt in &block.stmts {
            let span = match stmt.node {
                hir::StmtDecl(ref decl, _) => {
                    match decl.node {
                        hir::DeclLocal(_) => decl.span,

                        // Item statements are allowed
                        hir::DeclItem(_) => continue
                    }
                }
                hir::StmtExpr(ref expr, _) => expr.span,
                hir::StmtSemi(ref semi, _) => semi.span,
            };
            self.add_qualif(ConstQualif::NOT_CONST);
            if self.mode != Mode::Var {
                span_err!(self.tcx.sess, span, E0016,
                          "blocks in {}s are limited to items and \
                           tail expressions", self.msg());
            }
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
                    self.visit_expr(&**arg)
                }

                let inner = self.qualif;
                self.visit_expr(&**callee);
                // The callee's size doesn't count in the call.
                let added = self.qualif - inner;
                self.qualif = inner | (added - ConstQualif::NON_ZERO_SIZED);
            }
            hir::ExprRepeat(ref element, _) => {
                self.visit_expr(&**element);
                // The count is checked elsewhere (typeck).
                let count = match node_ty.sty {
                    ty::TyArray(_, n) => n,
                    _ => unreachable!()
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
            // Division by zero and overflow checking.
            hir::ExprBinary(op, _, _) => {
                intravisit::walk_expr(self, ex);
                let div_or_rem = op.node == hir::BiDiv || op.node == hir::BiRem;
                match node_ty.sty {
                    ty::TyUint(_) | ty::TyInt(_) if div_or_rem => {
                        if !self.qualif.intersects(ConstQualif::NOT_CONST) {
                            match const_eval::eval_const_expr_partial(
                                    self.tcx, ex, ExprTypeChecked, None) {
                                Ok(_) => {}
                                Err(ConstEvalErr { kind: IndexOpFeatureGated, ..}) => {},
                                Err(msg) => {
                                    self.tcx.sess.add_lint(::lint::builtin::CONST_ERR, ex.id,
                                                           msg.span,
                                                           msg.description().into_owned())
                                }
                            }
                        }
                    }
                    _ => {}
                }
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
                    if self.mode != Mode::Var {
                        span_err!(self.tcx.sess, ex.span, E0492,
                                  "cannot borrow a constant which contains \
                                   interior mutability, create a static instead");
                    }
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
                    } else {
                        span_err!(self.tcx.sess, ex.span, E0017,
                            "references in {}s may only refer \
                             to immutable values", self.msg())
                    }
                }
                if !self.qualif.intersects(ConstQualif::NON_STATIC_BORROWS) {
                    self.add_qualif(ConstQualif::HAS_STATIC_BORROWS);
                }
            }
            None => {}
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
            if v.mode != Mode::Var {
                span_err!(v.tcx.sess, e.span, E0493,
                          "{}s are not allowed to have destructors",
                          v.msg());
            }
        }
        _ => {}
    }

    let method_call = ty::MethodCall::expr(e.id);
    match e.node {
        hir::ExprUnary(..) |
        hir::ExprBinary(..) |
        hir::ExprIndex(..) if v.tcx.tables.borrow().method_map.contains_key(&method_call) => {
            v.add_qualif(ConstQualif::NOT_CONST);
            if v.mode != Mode::Var {
                span_err!(v.tcx.sess, e.span, E0011,
                            "user-defined operators are not allowed in {}s", v.msg());
            }
        }
        hir::ExprBox(_) => {
            v.add_qualif(ConstQualif::NOT_CONST);
            if v.mode != Mode::Var {
                span_err!(v.tcx.sess, e.span, E0010,
                          "allocations are not allowed in {}s", v.msg());
            }
        }
        hir::ExprUnary(op, ref inner) => {
            match v.tcx.node_id_to_type(inner.id).sty {
                ty::TyRawPtr(_) => {
                    assert!(op == hir::UnDeref);

                    v.add_qualif(ConstQualif::NOT_CONST);
                    if v.mode != Mode::Var {
                        span_err!(v.tcx.sess, e.span, E0396,
                                  "raw pointers cannot be dereferenced in {}s", v.msg());
                    }
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
                    if v.mode != Mode::Var {
                        span_err!(v.tcx.sess, e.span, E0395,
                                  "raw pointers cannot be compared in {}s", v.msg());
                    }
                }
                _ => {}
            }
        }
        hir::ExprCast(ref from, _) => {
            debug!("Checking const cast(id={})", from.id);
            match v.tcx.cast_kinds.borrow().get(&from.id) {
                None => v.tcx.sess.span_bug(e.span, "no kind for cast"),
                Some(&CastKind::PtrAddrCast) | Some(&CastKind::FnPtrAddrCast) => {
                    v.add_qualif(ConstQualif::NOT_CONST);
                    if v.mode != Mode::Var {
                        span_err!(v.tcx.sess, e.span, E0018,
                                  "raw pointers cannot be cast to integers in {}s", v.msg());
                    }
                }
                _ => {}
            }
        }
        hir::ExprPath(..) => {
            let def = v.tcx.def_map.borrow().get(&e.id).map(|d| d.full_def());
            match def {
                Some(def::DefVariant(_, _, _)) => {
                    // Count the discriminator or function pointer.
                    v.add_qualif(ConstQualif::NON_ZERO_SIZED);
                }
                Some(def::DefStruct(_)) => {
                    if let ty::TyBareFn(..) = node_ty.sty {
                        // Count the function pointer.
                        v.add_qualif(ConstQualif::NON_ZERO_SIZED);
                    }
                }
                Some(def::DefFn(..)) | Some(def::DefMethod(..)) => {
                    // Count the function pointer.
                    v.add_qualif(ConstQualif::NON_ZERO_SIZED);
                }
                Some(def::DefStatic(..)) => {
                    match v.mode {
                        Mode::Static | Mode::StaticMut => {}
                        Mode::Const | Mode::ConstFn => {
                            span_err!(v.tcx.sess, e.span, E0013,
                                "{}s cannot refer to other statics, insert \
                                 an intermediate constant instead", v.msg());
                        }
                        Mode::Var => v.add_qualif(ConstQualif::NOT_CONST)
                    }
                }
                Some(def::DefConst(did)) |
                Some(def::DefAssociatedConst(did)) => {
                    if let Some(expr) = const_eval::lookup_const_by_id(v.tcx, did,
                                                                       Some(e.id)) {
                        let inner = v.global_expr(Mode::Const, expr);
                        v.add_qualif(inner);
                    } else {
                        v.tcx.sess.span_bug(e.span,
                                            "DefConst or DefAssociatedConst \
                                             doesn't point to a constant");
                    }
                }
                Some(def::DefLocal(..)) if v.mode == Mode::ConstFn => {
                    // Sadly, we can't determine whether the types are zero-sized.
                    v.add_qualif(ConstQualif::NOT_CONST | ConstQualif::NON_ZERO_SIZED);
                }
                def => {
                    v.add_qualif(ConstQualif::NOT_CONST);
                    if v.mode != Mode::Var {
                        debug!("(checking const) found bad def: {:?}", def);
                        span_err!(v.tcx.sess, e.span, E0014,
                                  "paths in {}s may only refer to constants \
                                   or functions", v.msg());
                    }
                }
            }
        }
        hir::ExprCall(ref callee, _) => {
            let mut callee = &**callee;
            loop {
                callee = match callee.node {
                    hir::ExprBlock(ref block) => match block.expr {
                        Some(ref tail) => &**tail,
                        None => break
                    },
                    _ => break
                };
            }
            let def = v.tcx.def_map.borrow().get(&callee.id).map(|d| d.full_def());
            let is_const = match def {
                Some(def::DefStruct(..)) => true,
                Some(def::DefVariant(..)) => {
                    // Count the discriminator.
                    v.add_qualif(ConstQualif::NON_ZERO_SIZED);
                    true
                }
                Some(def::DefFn(did, _)) => {
                    v.handle_const_fn_call(e, did, node_ty)
                }
                Some(def::DefMethod(did)) => {
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
                if v.mode != Mode::Var {
                    // FIXME(#24111) Remove this check when const fn stabilizes
                    let (msg, note) =
                        if let UnstableFeatures::Disallow = v.tcx.sess.opts.unstable_features {
                        (format!("function calls in {}s are limited to \
                                  struct and enum constructors",
                                 v.msg()),
                         Some("a limited form of compile-time function \
                               evaluation is available on a nightly \
                               compiler via `const fn`"))
                    } else {
                        (format!("function calls in {}s are limited \
                                  to constant functions, \
                                  struct and enum constructors",
                                 v.msg()),
                         None)
                    };
                    let mut err = struct_span_err!(v.tcx.sess, e.span, E0015, "{}", msg);
                    if let Some(note) = note {
                        err.span_note(e.span, note);
                    }
                    err.emit();
                }
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
                if v.mode != Mode::Var {
                    span_err!(v.tcx.sess, e.span, E0378,
                              "method calls in {}s are limited to \
                               constant inherent methods", v.msg());
                }
            }
        }
        hir::ExprStruct(..) => {
            let did = v.tcx.def_map.borrow().get(&e.id).map(|def| def.def_id());
            if did == v.tcx.lang_items.unsafe_cell_type() {
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

        // Miscellaneous expressions that could be implemented.
        hir::ExprRange(..) |

        // Expressions with side-effects.
        hir::ExprAssign(..) |
        hir::ExprAssignOp(..) |
        hir::ExprInlineAsm(_) => {
            v.add_qualif(ConstQualif::NOT_CONST);
            if v.mode != Mode::Var {
                span_err!(v.tcx.sess, e.span, E0019,
                          "{} contains unimplemented expression type", v.msg());
            }
        }
    }
}

/// Check the adjustments of an expression
fn check_adjustments<'a, 'tcx>(v: &mut CheckCrateVisitor<'a, 'tcx>, e: &hir::Expr) {
    match v.tcx.tables.borrow().adjustments.get(&e.id) {
        None |
        Some(&ty::adjustment::AdjustReifyFnPointer) |
        Some(&ty::adjustment::AdjustUnsafeFnPointer) => {}

        Some(&ty::adjustment::AdjustDerefRef(
            ty::adjustment::AutoDerefRef { autoderefs, .. }
        )) => {
            if (0..autoderefs as u32).any(|autoderef| {
                    v.tcx.is_overloaded_autoderef(e.id, autoderef)
            }) {
                v.add_qualif(ConstQualif::NOT_CONST);
                if v.mode != Mode::Var {
                    span_err!(v.tcx.sess, e.span, E0400,
                              "user-defined dereference operators are not allowed in {}s",
                              v.msg());
                }
            }
        }
    }
}

pub fn check_crate(tcx: &ty::ctxt) {
    tcx.visit_all_items_in_krate(DepNode::CheckConst, &mut CheckCrateVisitor {
        tcx: tcx,
        mode: Mode::Var,
        qualif: ConstQualif::NOT_CONST,
        rvalue_borrows: NodeMap()
    });
    tcx.sess.abort_if_errors();
}

impl<'a, 'tcx> euv::Delegate<'tcx> for CheckCrateVisitor<'a, 'tcx> {
    fn consume(&mut self,
               _consume_id: ast::NodeId,
               consume_span: Span,
               cmt: mc::cmt,
               _mode: euv::ConsumeMode) {
        let mut cur = &cmt;
        loop {
            match cur.cat {
                Categorization::StaticItem => {
                    if self.mode != Mode::Var {
                        // statics cannot be consumed by value at any time, that would imply
                        // that they're an initializer (what a const is for) or kept in sync
                        // over time (not feasible), so deny it outright.
                        span_err!(self.tcx.sess, consume_span, E0394,
                                  "cannot refer to other statics by value, use the \
                                   address-of operator or a constant instead");
                    }
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
              borrow_span: Span,
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
        let mut is_interior = false;
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
                    if is_interior && self.mode != Mode::Var {
                        // Borrowed statics can specifically *only* have their address taken,
                        // not any number of other borrows such as borrowing fields, reading
                        // elements of an array, etc.
                        span_err!(self.tcx.sess, borrow_span, E0494,
                                  "cannot refer to the interior of another \
                                   static, use a constant instead");
                    }
                    break;
                }
                Categorization::Deref(ref cmt, _, _) |
                Categorization::Downcast(ref cmt, _) |
                Categorization::Interior(ref cmt, _) => {
                    is_interior = true;
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
