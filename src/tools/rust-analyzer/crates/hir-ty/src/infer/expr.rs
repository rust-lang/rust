//! Type inference for expressions.

use std::{iter::repeat_with, mem};

use either::Either;
use hir_def::{
    AdtId, FieldId, TupleFieldId, TupleId, VariantId,
    expr_store::path::{GenericArgs as HirGenericArgs, Path},
    hir::{
        Array, AsmOperand, AsmOptions, BinaryOp, BindingAnnotation, Expr, ExprId, ExprOrPatId,
        InlineAsmKind, LabelId, Pat, PatId, RecordLitField, RecordSpread, Statement, UnaryOp,
    },
    resolver::ValueNs,
    signatures::VariantFields,
};
use hir_def::{FunctionId, hir::ClosureKind};
use hir_expand::name::Name;
use rustc_ast_ir::Mutability;
use rustc_hash::FxHashMap;
use rustc_type_ir::{
    InferTy, Interner,
    inherent::{GenericArgs as _, IntoKind, Ty as _},
};
use stdx::never;
use syntax::ast::RangeOp;
use tracing::debug;

use crate::{
    Adjust, Adjustment, CallableDefId, Rawness, Span,
    consteval::literal_ty,
    infer::{AllowTwoPhase, BreakableKind, coerce::CoerceMany, find_continuable, pat::PatOrigin},
    lower::lower_mutability,
    method_resolution::{self, CandidateId, MethodCallee, MethodError},
    next_solver::{
        ClauseKind, FnSig, GenericArg, GenericArgs, Ty, TyKind, TypeError,
        infer::{
            BoundRegionConversionTime, InferOk,
            traits::{Obligation, ObligationCause},
        },
        obligation_ctxt::ObligationCtxt,
    },
};

use super::{
    BreakableContext, Diverges, Expectation, InferenceContext, InferenceDiagnostic,
    cast::CastCheck, find_breakable,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExprIsRead {
    Yes,
    No,
}

impl<'db> InferenceContext<'_, 'db> {
    pub(crate) fn infer_expr(
        &mut self,
        tgt_expr: ExprId,
        expected: &Expectation<'db>,
        is_read: ExprIsRead,
    ) -> Ty<'db> {
        let ty = self.infer_expr_inner(tgt_expr, expected, is_read);
        if let Some(expected_ty) = expected.only_has_type(&mut self.table) {
            _ = self.demand_eqtype(tgt_expr.into(), expected_ty, ty);
        }
        ty
    }

    pub(crate) fn infer_expr_suptype_coerce_never(
        &mut self,
        expr: ExprId,
        expected: &Expectation<'db>,
        is_read: ExprIsRead,
    ) -> Ty<'db> {
        let ty = self.infer_expr_inner(expr, expected, is_read);
        if ty.is_never() {
            if let Some(adjustments) = self.result.expr_adjustments.get(&expr) {
                return if let [Adjustment { kind: Adjust::NeverToAny, target }] = &**adjustments {
                    target.as_ref()
                } else {
                    self.err_ty()
                };
            }

            if let Some(target) = expected.only_has_type(&mut self.table) {
                self.coerce(expr, ty, target, AllowTwoPhase::No, ExprIsRead::Yes)
                    .expect("never-to-any coercion should always succeed")
            } else {
                ty
            }
        } else {
            if let Some(expected_ty) = expected.only_has_type(&mut self.table) {
                _ = self.demand_suptype(expr.into(), expected_ty, ty);
            }
            ty
        }
    }

    pub(crate) fn infer_expr_no_expect(
        &mut self,
        tgt_expr: ExprId,
        is_read: ExprIsRead,
    ) -> Ty<'db> {
        self.infer_expr_inner(tgt_expr, &Expectation::None, is_read)
    }

    /// Infer type of expression with possibly implicit coerce to the expected type.
    /// Return the type after possible coercion.
    pub(super) fn infer_expr_coerce(
        &mut self,
        expr: ExprId,
        expected: &Expectation<'db>,
        is_read: ExprIsRead,
    ) -> Ty<'db> {
        let ty = self.infer_expr_inner(expr, expected, is_read);
        if let Some(target) = expected.only_has_type(&mut self.table) {
            match self.coerce(expr, ty, target, AllowTwoPhase::No, is_read) {
                Ok(res) => res,
                Err(_) => {
                    self.emit_type_mismatch(expr.into(), target, ty);
                    target
                }
            }
        } else {
            ty
        }
    }

    /// Whether this expression constitutes a read of value of the type that
    /// it evaluates to.
    ///
    /// This is used to determine if we should consider the block to diverge
    /// if the expression evaluates to `!`, and if we should insert a `NeverToAny`
    /// coercion for values of type `!`.
    ///
    /// This function generally returns `false` if the expression is a place
    /// expression and the *parent* expression is the scrutinee of a match or
    /// the pointee of an `&` addr-of expression, since both of those parent
    /// expressions take a *place* and not a value.
    pub(super) fn expr_guaranteed_to_constitute_read_for_never(
        &mut self,
        expr: ExprId,
        is_read: ExprIsRead,
    ) -> bool {
        // rustc does the place expr check first, but since we are feeding
        // readness of the `expr` as a given value, we just can short-circuit
        // the place expr check if it's true(see codes and comments below)
        if is_read == ExprIsRead::Yes {
            return true;
        }

        // We only care about place exprs. Anything else returns an immediate
        // which would constitute a read. We don't care about distinguishing
        // "syntactic" place exprs since if the base of a field projection is
        // not a place then it would've been UB to read from it anyways since
        // that constitutes a read.
        if !self.is_syntactic_place_expr(expr) {
            return true;
        }

        // rustc queries parent hir node of `expr` here and determine whether
        // the current `expr` is read of value per its parent.
        // But since we don't have hir node, we cannot follow such "bottom-up"
        // method.
        // So, we pass down such readness from the parent expression through the
        // recursive `infer_expr*` calls in a "top-down" manner.
        is_read == ExprIsRead::Yes
    }

    /// Whether this pattern constitutes a read of value of the scrutinee that
    /// it is matching against. This is used to determine whether we should
    /// perform `NeverToAny` coercions.
    fn pat_guaranteed_to_constitute_read_for_never(&self, pat: PatId) -> bool {
        match &self.store[pat] {
            // Does not constitute a read.
            Pat::Wild | Pat::Rest => false,

            // This is unnecessarily restrictive when the pattern that doesn't
            // constitute a read is unreachable.
            //
            // For example `match *never_ptr { value => {}, _ => {} }` or
            // `match *never_ptr { _ if false => {}, value => {} }`.
            //
            // It is however fine to be restrictive here; only returning `true`
            // can lead to unsoundness.
            Pat::Or(subpats) => {
                subpats.iter().all(|pat| self.pat_guaranteed_to_constitute_read_for_never(*pat))
            }

            // All of these constitute a read, or match on something that isn't `!`,
            // which would require a `NeverToAny` coercion.
            Pat::Bind { .. }
            | Pat::TupleStruct { .. }
            | Pat::Path(_)
            | Pat::Tuple { .. }
            | Pat::Box { .. }
            | Pat::Deref { .. }
            | Pat::Ref { .. }
            | Pat::Lit(_)
            | Pat::Range { .. }
            | Pat::Slice { .. }
            | Pat::ConstBlock(_)
            | Pat::Record { .. }
            | Pat::NotNull
            | Pat::Missing => true,
            Pat::Expr(_) => unreachable!(
                "we don't call pat_guaranteed_to_constitute_read_for_never() with assignments"
            ),
        }
    }

    /// Checks if the pattern contains any `ref` or `ref mut` bindings, and if
    /// yes whether it contains mutable or just immutables ones.
    //
    // FIXME(tschottdorf): this is problematic as the HIR is being scraped, but
    // ref bindings are be implicit after #42640 (default match binding modes). See issue #44848.
    fn contains_explicit_ref_binding(&self, pat: PatId) -> bool {
        if let Pat::Bind { id, .. } = self.store[pat]
            && matches!(self.store[id].mode, BindingAnnotation::Ref | BindingAnnotation::RefMut)
        {
            return true;
        }

        let mut result = false;
        self.store.walk_pats_shallow(pat, |pat| result |= self.contains_explicit_ref_binding(pat));
        result
    }

    fn is_syntactic_place_expr(&self, expr: ExprId) -> bool {
        match &self.store[expr] {
            // Lang item paths cannot currently be local variables or statics.
            Expr::Path(Path::LangItem(_, _)) => false,
            Expr::Path(Path::Normal(path)) => path.type_anchor.is_none(),
            Expr::Path(path) => self
                .resolver
                .resolve_path_in_value_ns_fully(self.db, path, self.store.expr_path_hygiene(expr))
                .is_none_or(|res| matches!(res, ValueNs::LocalBinding(_) | ValueNs::StaticId(_))),
            Expr::Underscore => true,
            Expr::UnaryOp { op: UnaryOp::Deref, .. } => true,
            Expr::Field { .. } | Expr::Index { .. } => true,
            Expr::Call { .. }
            | Expr::MethodCall { .. }
            | Expr::Tuple { .. }
            | Expr::If { .. }
            | Expr::Match { .. }
            | Expr::Closure { .. }
            | Expr::Block { .. }
            | Expr::Array(..)
            | Expr::Break { .. }
            | Expr::Continue { .. }
            | Expr::Return { .. }
            | Expr::Become { .. }
            | Expr::Let { .. }
            | Expr::Loop { .. }
            | Expr::InlineAsm(..)
            | Expr::OffsetOf(..)
            | Expr::Literal(..)
            | Expr::Const(..)
            | Expr::UnaryOp { .. }
            | Expr::BinaryOp { .. }
            | Expr::Assignment { .. }
            | Expr::Yield { .. }
            | Expr::Cast { .. }
            | Expr::Unsafe { .. }
            | Expr::Await { .. }
            | Expr::Ref { .. }
            | Expr::Range { .. }
            | Expr::Box { .. }
            | Expr::RecordLit { .. }
            | Expr::Yeet { .. }
            | Expr::Missing => false,
        }
    }

    pub(crate) fn check_lhs_assignable(&self, lhs: ExprId) {
        if self.is_syntactic_place_expr(lhs) {
            return;
        }

        self.push_diagnostic(InferenceDiagnostic::InvalidLhsOfAssignment { lhs });
    }

    fn infer_expr_coerce_never(
        &mut self,
        expr: ExprId,
        expected: &Expectation<'db>,
        is_read: ExprIsRead,
    ) -> Ty<'db> {
        let ty = self.infer_expr_inner(expr, expected, is_read);
        // While we don't allow *arbitrary* coercions here, we *do* allow
        // coercions from `!` to `expected`.
        if ty.is_never() {
            if let Some(adjustments) = self.result.expr_adjustments.get(&expr) {
                return if let [Adjustment { kind: Adjust::NeverToAny, target }] = &**adjustments {
                    target.as_ref()
                } else {
                    self.err_ty()
                };
            }

            if let Some(target) = expected.only_has_type(&mut self.table) {
                self.coerce(expr, ty, target, AllowTwoPhase::No, ExprIsRead::Yes)
                    .expect("never-to-any coercion should always succeed")
            } else {
                ty
            }
        } else {
            if let Some(expected_ty) = expected.only_has_type(&mut self.table) {
                _ = self.demand_eqtype(expr.into(), ty, expected_ty);
            }
            ty
        }
    }

    #[tracing::instrument(level = "debug", skip(self, is_read), ret)]
    pub(super) fn infer_expr_inner(
        &mut self,
        tgt_expr: ExprId,
        expected: &Expectation<'db>,
        is_read: ExprIsRead,
    ) -> Ty<'db> {
        self.db.unwind_if_revision_cancelled();

        let expr = &self.store[tgt_expr];
        tracing::trace!(?expr);
        let ty = match expr {
            Expr::Missing => self.err_ty(),
            &Expr::If { condition, then_branch, else_branch } => {
                let expected = &expected.adjust_for_branches(&mut self.table, tgt_expr.into());
                self.infer_expr_coerce_never(
                    condition,
                    &Expectation::HasType(self.types.types.bool),
                    ExprIsRead::Yes,
                );

                let condition_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);

                let then_ty = self.infer_expr_inner(then_branch, expected, ExprIsRead::Yes);
                let then_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
                let mut coercion_sites = [then_branch, tgt_expr];
                if let Some(else_branch) = else_branch {
                    coercion_sites[1] = else_branch;
                }
                let mut coerce = CoerceMany::with_coercion_sites(
                    expected.coercion_target_type(&mut self.table, then_branch.into()),
                    &coercion_sites,
                );
                coerce.coerce(
                    self,
                    &ObligationCause::new(then_branch),
                    then_branch,
                    then_ty,
                    ExprIsRead::Yes,
                );
                match else_branch {
                    Some(else_branch) => {
                        let else_ty = self.infer_expr_inner(else_branch, expected, ExprIsRead::Yes);
                        let else_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
                        coerce.coerce(
                            self,
                            &ObligationCause::new(else_branch),
                            else_branch,
                            else_ty,
                            ExprIsRead::Yes,
                        );
                        self.diverges = condition_diverges | then_diverges & else_diverges;
                    }
                    None => {
                        coerce.coerce_forced_unit(
                            self,
                            tgt_expr,
                            &ObligationCause::new(tgt_expr),
                            true,
                            ExprIsRead::Yes,
                        );
                        self.diverges = condition_diverges;
                    }
                }

                coerce.complete(self)
            }
            &Expr::Let { pat, expr } => {
                let child_is_read = if self.pat_guaranteed_to_constitute_read_for_never(pat) {
                    ExprIsRead::Yes
                } else {
                    ExprIsRead::No
                };
                let input_ty = self.infer_expr(expr, &Expectation::none(), child_is_read);
                self.infer_top_pat(pat, input_ty, PatOrigin::LetExpr);
                self.types.types.bool
            }
            Expr::Block { statements, tail, label, id: _ } => {
                self.infer_block(tgt_expr, statements, *tail, *label, expected)
            }
            Expr::Unsafe { id: _, statements, tail } => {
                self.infer_block(tgt_expr, statements, *tail, None, expected)
            }
            Expr::Const(id) => {
                self.with_breakable_ctx(BreakableKind::Border, None, None, |this| {
                    this.infer_expr(*id, expected, ExprIsRead::Yes)
                })
                .1
            }
            &Expr::Loop { body, label } => {
                let ty = expected.coercion_target_type(&mut self.table, tgt_expr.into());
                let (breaks, ()) =
                    self.with_breakable_ctx(BreakableKind::Loop, Some(ty), label, |this| {
                        this.infer_expr(
                            body,
                            &Expectation::HasType(this.types.types.unit),
                            ExprIsRead::Yes,
                        );
                    });

                match breaks {
                    Some(breaks) => {
                        self.diverges = Diverges::Maybe;
                        breaks
                    }
                    None => self.types.types.never,
                }
            }
            Expr::Closure { body, args, ret_type, arg_types, closure_kind, capture_by: _ } => self
                .infer_closure(
                    *body,
                    args,
                    *ret_type,
                    arg_types,
                    *closure_kind,
                    tgt_expr,
                    expected,
                ),
            Expr::Call { callee, args, .. } => self.infer_call(tgt_expr, *callee, args, expected),
            Expr::MethodCall { receiver, args, method_name, generic_args } => self
                .infer_method_call(
                    tgt_expr,
                    *receiver,
                    args,
                    method_name,
                    generic_args.as_deref(),
                    expected,
                ),
            Expr::Match { expr, arms } => {
                let mut scrutinee_is_read = true;
                let mut contains_ref_bindings = false;
                for arm in arms {
                    scrutinee_is_read &= self.pat_guaranteed_to_constitute_read_for_never(arm.pat);
                    contains_ref_bindings |= self.contains_explicit_ref_binding(arm.pat);
                }
                let scrutinee_is_read =
                    if scrutinee_is_read { ExprIsRead::Yes } else { ExprIsRead::No };
                let input_ty = self.demand_scrutinee_type(
                    *expr,
                    contains_ref_bindings,
                    arms.is_empty(),
                    scrutinee_is_read,
                );

                if arms.is_empty() {
                    self.diverges = Diverges::Always;
                    self.types.types.never
                } else {
                    let matchee_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
                    let mut all_arms_diverge = Diverges::Always;
                    for arm in arms.iter() {
                        self.infer_top_pat(arm.pat, input_ty, PatOrigin::MatchArm);
                    }

                    let expected = expected.adjust_for_branches(&mut self.table, tgt_expr.into());
                    let result_ty = match &expected {
                        // We don't coerce to `()` so that if the match expression is a
                        // statement it's branches can have any consistent type.
                        Expectation::HasType(ty) if *ty != self.types.types.unit => *ty,
                        _ => self.table.next_ty_var((*expr).into()),
                    };
                    let mut coerce = CoerceMany::new(result_ty);

                    for arm in arms.iter() {
                        if let Some(guard_expr) = arm.guard {
                            self.diverges = Diverges::Maybe;
                            self.infer_expr_coerce_never(
                                guard_expr,
                                &Expectation::HasType(self.types.types.bool),
                                ExprIsRead::Yes,
                            );
                        }
                        self.diverges = Diverges::Maybe;

                        let arm_ty = self.infer_expr_inner(arm.expr, &expected, ExprIsRead::Yes);
                        all_arms_diverge &= self.diverges;
                        coerce.coerce(
                            self,
                            &ObligationCause::new(arm.expr),
                            arm.expr,
                            arm_ty,
                            ExprIsRead::Yes,
                        );
                    }

                    self.diverges = matchee_diverges | all_arms_diverge;

                    coerce.complete(self)
                }
            }
            Expr::Path(p) => self.infer_expr_path(p, tgt_expr.into(), tgt_expr),
            &Expr::Continue { label } => {
                if find_continuable(&mut self.breakables, label).is_none() {
                    self.push_diagnostic(InferenceDiagnostic::BreakOutsideOfLoop {
                        expr: tgt_expr,
                        is_break: false,
                        bad_value_break: false,
                    });
                };
                self.types.types.never
            }
            &Expr::Break { expr, label } => {
                let val_ty = if let Some(expr) = expr {
                    let opt_coerce_to = match find_breakable(&mut self.breakables, label) {
                        Some(ctxt) => match &ctxt.coerce {
                            Some(coerce) => coerce.expected_ty(),
                            None => {
                                self.push_diagnostic(InferenceDiagnostic::BreakOutsideOfLoop {
                                    expr: tgt_expr,
                                    is_break: true,
                                    bad_value_break: true,
                                });
                                self.err_ty()
                            }
                        },
                        None => self.err_ty(),
                    };
                    self.infer_expr_inner(
                        expr,
                        &Expectation::HasType(opt_coerce_to),
                        ExprIsRead::Yes,
                    )
                } else {
                    self.types.types.unit
                };

                match find_breakable(&mut self.breakables, label) {
                    Some(ctxt) => match ctxt.coerce.take() {
                        Some(mut coerce) => {
                            let expr = expr.unwrap_or(tgt_expr);
                            coerce.coerce(
                                self,
                                &ObligationCause::new(expr),
                                expr,
                                val_ty,
                                ExprIsRead::Yes,
                            );

                            // Avoiding borrowck
                            let ctxt = find_breakable(&mut self.breakables, label)
                                .expect("breakable stack changed during coercion");
                            ctxt.may_break = true;
                            ctxt.coerce = Some(coerce);
                        }
                        None => ctxt.may_break = true,
                    },
                    None => {
                        self.push_diagnostic(InferenceDiagnostic::BreakOutsideOfLoop {
                            expr: tgt_expr,
                            is_break: true,
                            bad_value_break: false,
                        });
                    }
                }
                self.types.types.never
            }
            &Expr::Return { expr } => self.infer_expr_return(tgt_expr, expr),
            &Expr::Become { expr } => self.infer_expr_become(expr),
            Expr::Yield { expr } => {
                if let Some((resume_ty, yield_ty)) = self.resume_yield_tys {
                    if let Some(expr) = expr {
                        self.infer_expr_coerce(
                            *expr,
                            &Expectation::has_type(yield_ty),
                            ExprIsRead::Yes,
                        );
                    } else {
                        let unit = self.types.types.unit;
                        let _ = self.coerce(
                            tgt_expr,
                            unit,
                            yield_ty,
                            AllowTwoPhase::No,
                            ExprIsRead::Yes,
                        );
                    }
                    resume_ty
                } else {
                    // FIXME: report error (yield expr in non-coroutine)
                    self.types.types.error
                }
            }
            Expr::Yeet { expr } => {
                if let &Some(expr) = expr {
                    self.infer_expr_no_expect(expr, ExprIsRead::Yes);
                }
                self.types.types.never
            }
            Expr::RecordLit { path, fields, spread, .. } => {
                self.infer_record_expr(tgt_expr, expected, path, fields, *spread)
            }
            Expr::Field { expr, name } => self.infer_field_access(tgt_expr, *expr, name, expected),
            Expr::Await { expr } => self.infer_await_expr(tgt_expr, *expr),
            Expr::Cast { expr, type_ref } => {
                let cast_ty = self.make_body_ty(*type_ref);
                let expr_ty =
                    self.infer_expr(*expr, &Expectation::Castable(cast_ty), ExprIsRead::Yes);
                self.deferred_cast_checks.push(CastCheck::new(tgt_expr, *expr, expr_ty, cast_ty));
                cast_ty
            }
            Expr::Ref { expr, rawness, mutability } => self.infer_ref_expr(
                *rawness,
                lower_mutability(*mutability),
                *expr,
                expected,
                tgt_expr,
            ),
            &Expr::Box { expr } => self.infer_expr_box(expr, expected),
            Expr::UnaryOp { expr, op } => self.infer_unop_expr(*op, *expr, expected, tgt_expr),
            Expr::BinaryOp { lhs, rhs, op } => match op {
                Some(BinaryOp::Assignment { op: Some(op) }) => {
                    self.infer_assign_op_expr(tgt_expr, *op, *lhs, *rhs)
                }
                Some(op) => self.infer_binop_expr(tgt_expr, *op, *lhs, *rhs),
                None => self.err_ty(),
            },
            &Expr::Assignment { target, value } => {
                // In ordinary (non-destructuring) assignments, the type of
                // `lhs` must be inferred first so that the ADT fields
                // instantiations in RHS can be coerced to it. Note that this
                // cannot happen in destructuring assignments because of how
                // they are desugared.
                let lhs_ty = match &self.store[target] {
                    // LHS of assignment doesn't constitute reads.
                    &Pat::Expr(expr) => {
                        Some(self.infer_expr(expr, &Expectation::none(), ExprIsRead::No))
                    }
                    _ => None,
                };
                let is_destructuring_assignment = lhs_ty.is_none();

                if let Some(lhs_ty) = lhs_ty {
                    self.write_pat_ty(target, lhs_ty);
                    self.infer_expr_coerce(value, &Expectation::has_type(lhs_ty), ExprIsRead::Yes);
                } else {
                    let rhs_ty = self.infer_expr(value, &Expectation::none(), ExprIsRead::Yes);
                    let resolver_guard =
                        self.resolver.update_to_inner_scope(self.db, self.store_owner, tgt_expr);
                    self.inside_assignment = true;
                    self.infer_top_pat(target, rhs_ty, PatOrigin::DestructuringAssignment);
                    self.inside_assignment = false;
                    self.resolver.reset_to_guard(resolver_guard);
                }
                if is_destructuring_assignment && self.diverges.is_always() {
                    // Ordinary assignments always return `()`, even when they diverge.
                    // However, rustc lowers destructuring assignments into blocks, and blocks return `!` if they have no tail
                    // expression and they diverge. Therefore, we have to do the same here, even though we don't lower destructuring
                    // assignments into blocks.
                    self.table.new_maybe_never_var(value.into())
                } else {
                    self.types.types.unit
                }
            }
            Expr::Range { lhs, rhs, range_type } => {
                let lhs_ty =
                    lhs.map(|e| self.infer_expr_inner(e, &Expectation::none(), ExprIsRead::Yes));
                let rhs_expect = lhs_ty.map_or_else(Expectation::none, Expectation::has_type);
                let rhs_ty = rhs.map(|e| self.infer_expr(e, &rhs_expect, ExprIsRead::Yes));
                let single_arg_adt = |adt, ty: Ty<'db>| {
                    Ty::new_adt(
                        self.interner(),
                        adt,
                        GenericArgs::new_from_slice(&[GenericArg::from(ty)]),
                    )
                };
                match (range_type, lhs_ty, rhs_ty) {
                    (RangeOp::Exclusive, None, None) => match self.resolve_range_full() {
                        Some(adt) => {
                            Ty::new_adt(self.interner(), adt, self.types.empty.generic_args)
                        }
                        None => self.err_ty(),
                    },
                    (RangeOp::Exclusive, None, Some(ty)) => match self.resolve_range_to() {
                        Some(adt) => single_arg_adt(adt, ty),
                        None => self.err_ty(),
                    },
                    (RangeOp::Inclusive, None, Some(ty)) => {
                        match self.resolve_range_to_inclusive() {
                            Some(adt) => single_arg_adt(adt, ty),
                            None => self.err_ty(),
                        }
                    }
                    (RangeOp::Exclusive, Some(_), Some(ty)) => match self.resolve_range() {
                        Some(adt) => single_arg_adt(adt, ty),
                        None => self.err_ty(),
                    },
                    (RangeOp::Inclusive, Some(_), Some(ty)) => {
                        match self.resolve_range_inclusive() {
                            Some(adt) => single_arg_adt(adt, ty),
                            None => self.err_ty(),
                        }
                    }
                    (RangeOp::Exclusive, Some(ty), None) => match self.resolve_range_from() {
                        Some(adt) => single_arg_adt(adt, ty),
                        None => self.err_ty(),
                    },
                    (RangeOp::Inclusive, _, None) => self.err_ty(),
                }
            }
            Expr::Index { base, index } => {
                let base_t = self.infer_expr_no_expect(*base, ExprIsRead::Yes);
                let idx_t = self.infer_expr_no_expect(*index, ExprIsRead::Yes);

                let base_t = self.structurally_resolve_type((*base).into(), base_t);
                match self.lookup_indexing(tgt_expr, *base, *index, base_t, idx_t) {
                    Some((trait_index_ty, trait_element_ty)) => {
                        // two-phase not needed because index_ty is never mutable
                        self.demand_coerce(
                            *index,
                            idx_t,
                            trait_index_ty,
                            AllowTwoPhase::No,
                            ExprIsRead::Yes,
                        );
                        self.table.select_obligations_where_possible();
                        trait_element_ty
                    }
                    // FIXME: Report an error.
                    None => self.types.types.error,
                }
            }
            Expr::Tuple { exprs, .. } => {
                let mut tys = match expected
                    .only_has_type(&mut self.table)
                    .map(|t| self.table.try_structurally_resolve_type(tgt_expr.into(), t).kind())
                {
                    Some(TyKind::Tuple(substs)) => substs
                        .iter()
                        .chain(repeat_with(|| self.table.next_ty_var(Span::Dummy)))
                        .take(exprs.len())
                        .collect::<Vec<_>>(),
                    _ => exprs.iter().map(|&expr| self.table.next_ty_var(expr.into())).collect(),
                };

                for (expr, ty) in exprs.iter().zip(tys.iter_mut()) {
                    *ty =
                        self.infer_expr_coerce(*expr, &Expectation::has_type(*ty), ExprIsRead::Yes);
                }

                Ty::new_tup(self.interner(), &tys)
            }
            Expr::Array(Array::ElementList { elements }) => {
                self.infer_array_elements_expr(elements, expected, tgt_expr)
            }
            Expr::Array(Array::Repeat { initializer, repeat }) => {
                self.infer_array_repeat_expr(*initializer, *repeat, expected, tgt_expr)
            }
            Expr::Literal(lit) => literal_ty(
                self.interner(),
                lit,
                |_| {
                    let expected_ty = expected.to_option(&self.table);
                    tracing::debug!(?expected_ty);
                    let opt_ty = match expected_ty.as_ref().map(|it| it.kind()) {
                        Some(TyKind::Int(_) | TyKind::Uint(_)) => expected_ty,
                        Some(TyKind::Char) => Some(self.types.types.u8),
                        Some(TyKind::RawPtr(..) | TyKind::FnDef(..) | TyKind::FnPtr(..)) => {
                            Some(self.types.types.usize)
                        }
                        _ => None,
                    };
                    opt_ty.unwrap_or_else(|| self.table.next_int_var())
                },
                |_| {
                    let expected_ty = expected.to_option(&self.table);
                    let opt_ty = match expected_ty.as_ref().map(|it| it.kind()) {
                        Some(TyKind::Int(_) | TyKind::Uint(_)) => expected_ty,
                        Some(TyKind::Char) => Some(self.types.types.u8),
                        Some(TyKind::RawPtr(..) | TyKind::FnDef(..) | TyKind::FnPtr(..)) => {
                            Some(self.types.types.usize)
                        }
                        _ => None,
                    };
                    opt_ty.unwrap_or_else(|| self.table.next_int_var())
                },
                |_| {
                    let opt_ty = expected
                        .to_option(&self.table)
                        .filter(|ty| matches!(ty.kind(), TyKind::Float(_)));
                    opt_ty.unwrap_or_else(|| self.table.next_float_var())
                },
            ),
            Expr::Underscore => {
                // Underscore expression is an error, we render a specialized diagnostic
                // to let the user know what type is expected though.
                let expected = expected.to_option(&self.table).unwrap_or_else(|| self.err_ty());
                self.push_diagnostic(InferenceDiagnostic::TypedHole {
                    expr: tgt_expr,
                    expected: expected.store(),
                });
                expected
            }
            Expr::OffsetOf(_) => self.types.types.usize,
            Expr::InlineAsm(asm) => {
                let check_expr_asm_operand = |this: &mut Self, expr, is_input: bool| {
                    let ty = this.infer_expr_no_expect(expr, ExprIsRead::Yes);

                    // If this is an input value, we require its type to be fully resolved
                    // at this point. This allows us to provide helpful coercions which help
                    // pass the type candidate list in a later pass.
                    //
                    // We don't require output types to be resolved at this point, which
                    // allows them to be inferred based on how they are used later in the
                    // function.
                    if is_input {
                        let ty = this.structurally_resolve_type(expr.into(), ty);
                        match ty.kind() {
                            TyKind::FnDef(def, parameters) => {
                                let fnptr_ty = Ty::new_fn_ptr(
                                    this.interner(),
                                    this.interner()
                                        .fn_sig(def)
                                        .instantiate(this.interner(), parameters)
                                        .skip_norm_wip(),
                                );
                                _ = this.coerce(
                                    expr,
                                    ty,
                                    fnptr_ty,
                                    AllowTwoPhase::No,
                                    ExprIsRead::Yes,
                                );
                            }
                            TyKind::Ref(_, base_ty, mutbl) => {
                                let ptr_ty = Ty::new_ptr(this.interner(), base_ty, mutbl);
                                _ = this.coerce(
                                    expr,
                                    ty,
                                    ptr_ty,
                                    AllowTwoPhase::No,
                                    ExprIsRead::Yes,
                                );
                            }
                            _ => {}
                        }
                    }
                };

                let diverge = asm.options.contains(AsmOptions::NORETURN);
                asm.operands.iter().for_each(|(_, operand)| match *operand {
                    AsmOperand::In { expr, .. } => check_expr_asm_operand(self, expr, true),
                    AsmOperand::Out { expr: Some(expr), .. } | AsmOperand::InOut { expr, .. } => {
                        check_expr_asm_operand(self, expr, false)
                    }
                    AsmOperand::Out { expr: None, .. } => (),
                    AsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                        check_expr_asm_operand(self, in_expr, true);
                        if let Some(out_expr) = out_expr {
                            check_expr_asm_operand(self, out_expr, false);
                        }
                    }
                    AsmOperand::Label(expr) => {
                        self.infer_expr(
                            expr,
                            &Expectation::HasType(self.types.types.unit),
                            ExprIsRead::No,
                        );
                    }
                    AsmOperand::Const(expr) => {
                        self.infer_expr(expr, &Expectation::None, ExprIsRead::No);
                    }
                    // FIXME: `sym` should report for things that are not functions or statics.
                    AsmOperand::Sym(_) => (),
                });
                if diverge || asm.kind == InlineAsmKind::NakedAsm {
                    self.types.types.never
                } else {
                    self.types.types.unit
                }
            }
        };
        let ty = self.insert_type_vars_shallow(ty);
        self.write_expr_ty(tgt_expr, ty);
        if self.shallow_resolve(ty).is_never()
            && self.expr_guaranteed_to_constitute_read_for_never(tgt_expr, is_read)
        {
            // Any expression that produces a value of type `!` must have diverged
            self.diverges = Diverges::Always;
        }
        ty
    }

    fn infer_ref_expr(
        &mut self,
        rawness: Rawness,
        mutbl: Mutability,
        oprnd: ExprId,
        expected: &Expectation<'db>,
        expr: ExprId,
    ) -> Ty<'db> {
        let hint = expected.only_has_type(&mut self.table).map_or(Expectation::None, |ty| {
            match self.table.resolve_vars_with_obligations(ty).kind() {
                TyKind::Ref(_, ty, _) | TyKind::RawPtr(ty, _) => {
                    if self.is_syntactic_place_expr(oprnd) {
                        // Places may legitimately have unsized types.
                        // For example, dereferences of a wide pointer and
                        // the last field of a struct can be unsized.
                        Expectation::has_type(ty)
                    } else {
                        Expectation::rvalue_hint(self, ty)
                    }
                }
                _ => Expectation::None,
            }
        });
        let ty = self.infer_expr_inner(oprnd, &hint, ExprIsRead::No);

        match rawness {
            Rawness::RawPtr => Ty::new_ptr(self.interner(), ty, mutbl),
            Rawness::Ref => {
                // Note: at this point, we cannot say what the best lifetime
                // is to use for resulting pointer. We want to use the
                // shortest lifetime possible so as to avoid spurious borrowck
                // errors. Moreover, the longest lifetime will depend on the
                // precise details of the value whose address is being taken
                // (and how long it is valid), which we don't know yet until
                // type inference is complete.
                //
                // Therefore, here we simply generate a region variable. The
                // region inferencer will then select a suitable value.
                // Finally, borrowck will infer the value of the region again,
                // this time with enough precision to check that the value
                // whose address was taken can actually be made to live as long
                // as it needs to live.
                let region = self.table.next_region_var(expr.into());
                Ty::new_ref(self.interner(), region, ty, mutbl)
            }
        }
    }

    fn infer_await_expr(&mut self, expr: ExprId, awaitee: ExprId) -> Ty<'db> {
        let awaitee_ty = self.infer_expr_no_expect(awaitee, ExprIsRead::Yes);
        let (Some(into_future), Some(into_future_output)) =
            (self.lang_items.IntoFuture, self.lang_items.IntoFutureOutput)
        else {
            return self.types.types.error;
        };
        self.table.register_bound(awaitee_ty, into_future, ObligationCause::new(expr));
        // Do not eagerly normalize.
        Ty::new_projection(self.interner(), into_future_output.into(), [awaitee_ty])
    }

    fn infer_record_expr(
        &mut self,
        expr: ExprId,
        expected: &Expectation<'db>,
        path: &Path,
        fields: &[RecordLitField],
        base_expr: RecordSpread,
    ) -> Ty<'db> {
        // Find the relevant variant
        let (adt_ty, Some(variant)) = self.resolve_variant(expr.into(), path, false) else {
            // FIXME: Emit an error.
            for field in fields {
                self.infer_expr_no_expect(field.expr, ExprIsRead::Yes);
            }

            return self.types.types.error;
        };
        self.write_variant_resolution(expr.into(), variant);

        // Prohibit struct expressions when non-exhaustive flag is set.
        if self.has_applicable_non_exhaustive(variant.into()) {
            self.push_diagnostic(InferenceDiagnostic::NonExhaustiveRecordExpr { expr });
        }

        self.check_record_expr_fields(adt_ty, expected, expr, variant, fields, base_expr);

        self.require_type_is_sized(adt_ty, expr.into());
        adt_ty
    }

    fn check_record_expr_fields(
        &mut self,
        adt_ty: Ty<'db>,
        expected: &Expectation<'db>,
        expr: ExprId,
        variant: VariantId,
        hir_fields: &[RecordLitField],
        base_expr: RecordSpread,
    ) {
        let interner = self.interner();

        let adt_ty = self.table.try_structurally_resolve_type(expr.into(), adt_ty);
        let adt_ty_hint = expected.only_has_type(&mut self.table).and_then(|expected| {
            self.infcx()
                .fudge_inference_if_ok(|| {
                    let mut ocx = ObligationCtxt::new(self.infcx());
                    ocx.sup(&ObligationCause::new(expr), self.table.param_env, expected, adt_ty)?;
                    if !ocx.try_evaluate_obligations().is_empty() {
                        return Err(TypeError::Mismatch);
                    }
                    Ok(self.resolve_vars_if_possible(adt_ty))
                })
                .ok()
        });
        if let Some(adt_ty_hint) = adt_ty_hint {
            // re-link the variables that the fudging above can create.
            _ = self.demand_eqtype(expr.into(), adt_ty_hint, adt_ty);
        }

        let TyKind::Adt(adt, args) = adt_ty.kind() else {
            never!("non-ADT passed to check_struct_expr_fields");
            return;
        };
        let adt_id = adt.def_id();

        let variant_fields = variant.fields(self.db);
        let variant_field_tys = self.db.field_types(variant);
        let variant_field_vis = VariantFields::field_visibilities(self.db, variant);
        let mut remaining_fields = variant_fields
            .fields()
            .iter()
            .map(|(i, field)| (field.name.clone(), i))
            .collect::<FxHashMap<_, _>>();

        let mut seen_fields = FxHashMap::default();

        // Type-check each field.
        for field in hir_fields {
            let name = &field.name;
            let field_type = if let Some(i) = remaining_fields.remove(name) {
                seen_fields.insert(name, i);

                if !self.resolver.is_visible(self.db, variant_field_vis[i]) {
                    self.push_diagnostic(InferenceDiagnostic::NoSuchField {
                        field: field.expr.into(),
                        private: Some(i),
                        variant,
                    });
                }

                variant_field_tys[i].get().instantiate(interner, args).skip_norm_wip()
            } else {
                if let Some(field_idx) = seen_fields.get(&name) {
                    self.push_diagnostic(InferenceDiagnostic::DuplicateField {
                        field: field.expr.into(),
                        variant,
                    });
                    variant_field_tys[*field_idx].get().instantiate(interner, args).skip_norm_wip()
                } else {
                    self.push_diagnostic(InferenceDiagnostic::NoSuchField {
                        field: field.expr.into(),
                        private: None,
                        variant,
                    });
                    self.types.types.error
                }
            };

            // Check that the expected field type is WF. Otherwise, we emit no use-site error
            // in the case of coercions for non-WF fields, which leads to incorrect error
            // tainting. See issue #126272.
            self.table.register_wf_obligation(field_type.into(), ObligationCause::new(field.expr));

            // Make sure to give a type to the field even if there's
            // an error, so we can continue type-checking.
            self.infer_expr_coerce(field.expr, &Expectation::has_type(field_type), ExprIsRead::Yes);
        }

        // Make sure the programmer specified correct number of fields.
        if matches!(adt_id, AdtId::UnionId(_)) && hir_fields.len() != 1 {
            self.push_diagnostic(InferenceDiagnostic::UnionExprMustHaveExactlyOneField { expr });
        }

        match base_expr {
            RecordSpread::FieldDefaults => {
                let mut missing_mandatory_fields = Vec::new();
                let mut missing_optional_fields = Vec::new();
                for (field_idx, field) in variant_fields.fields().iter() {
                    if remaining_fields.remove(&field.name).is_some() {
                        if field.default_value.is_none() {
                            missing_mandatory_fields.push(field_idx);
                        } else {
                            missing_optional_fields.push(field_idx);
                        }
                    }
                }
                if !missing_mandatory_fields.is_empty() {
                    // FIXME: Emit an error: missing fields.
                }
            }
            RecordSpread::Expr(base_expr) => {
                // FIXME: We are currently creating two branches here in order to maintain
                // consistency. But they should be merged as much as possible.
                if self.features.type_changing_struct_update {
                    if matches!(adt_id, AdtId::StructId(_)) {
                        // Make some fresh generic parameters for our ADT type.
                        let fresh_args = self.table.fresh_args_for_item(expr.into(), adt_id.into());
                        // We do subtyping on the FRU fields first, so we can
                        // learn exactly what types we expect the base expr
                        // needs constrained to be compatible with the struct
                        // type we expect from the expectation value.
                        for (field_idx, field) in variant_fields.fields().iter() {
                            let fru_ty = variant_field_tys[field_idx]
                                .get()
                                .instantiate(interner, fresh_args)
                                .skip_norm_wip();
                            if remaining_fields.remove(&field.name).is_some() {
                                let target_ty = variant_field_tys[field_idx]
                                    .get()
                                    .instantiate(interner, args)
                                    .skip_norm_wip();
                                let cause = ObligationCause::new(expr);
                                match self.table.at(&cause).sup(target_ty, fru_ty) {
                                    Ok(InferOk { obligations, value: () }) => {
                                        self.table.register_predicates(obligations)
                                    }
                                    Err(_) => {
                                        never!(
                                            "subtyping remaining fields of type changing FRU \
                                                failed: {target_ty:?} != {fru_ty:?}: {:?}",
                                            field.name,
                                        );
                                    }
                                }
                            }
                        }
                        // The use of fresh args that we have subtyped against
                        // our base ADT type's fields allows us to guide inference
                        // along so that, e.g.
                        // ```
                        // MyStruct<'a, F1, F2, const C: usize> {
                        //     f: F1,
                        //     // Other fields that reference `'a`, `F2`, and `C`
                        // }
                        //
                        // let x = MyStruct {
                        //    f: 1usize,
                        //    ..other_struct
                        // };
                        // ```
                        // will have the `other_struct` expression constrained to
                        // `MyStruct<'a, _, F2, C>`, as opposed to just `_`...
                        // This is important to allow coercions to happen in
                        // `other_struct` itself. See `coerce-in-base-expr.rs`.
                        let fresh_base_ty = Ty::new_adt(self.interner(), adt_id, fresh_args);
                        self.infer_expr_suptype_coerce_never(
                            base_expr,
                            &Expectation::has_type(self.resolve_vars_if_possible(fresh_base_ty)),
                            ExprIsRead::Yes,
                        );
                    } else {
                        // Check the base_expr, regardless of a bad expected adt_ty, so we can get
                        // type errors on that expression, too.
                        self.infer_expr_no_expect(base_expr, ExprIsRead::Yes);
                        self.push_diagnostic(
                            InferenceDiagnostic::FunctionalRecordUpdateOnNonStruct { base_expr },
                        );
                    }
                } else {
                    self.infer_expr_suptype_coerce_never(
                        base_expr,
                        &Expectation::has_type(adt_ty),
                        ExprIsRead::Yes,
                    );
                    if !matches!(adt_id, AdtId::StructId(_)) {
                        self.push_diagnostic(
                            InferenceDiagnostic::FunctionalRecordUpdateOnNonStruct { base_expr },
                        );
                    }
                }
            }
            RecordSpread::None => {
                if !matches!(adt_id, AdtId::UnionId(_))
                    && !remaining_fields.is_empty()
                    //~ non_exhaustive already reported, which will only happen for extern modules
                    && !self.has_applicable_non_exhaustive(adt_id.into())
                {
                    debug!(?remaining_fields);

                    // FIXME: Emit an error: missing fields.
                }
            }
        }
    }

    fn demand_scrutinee_type(
        &mut self,
        scrut: ExprId,
        contains_ref_bindings: bool,
        no_arms: bool,
        scrutinee_is_read: ExprIsRead,
    ) -> Ty<'db> {
        // Not entirely obvious: if matches may create ref bindings, we want to
        // use the *precise* type of the scrutinee, *not* some supertype, as
        // the "scrutinee type" (issue #23116).
        //
        // arielb1 [writes here in this comment thread][c] that there
        // is certainly *some* potential danger, e.g., for an example
        // like:
        //
        // [c]: https://github.com/rust-lang/rust/pull/43399#discussion_r130223956
        //
        // ```
        // let Foo(x) = f()[0];
        // ```
        //
        // Then if the pattern matches by reference, we want to match
        // `f()[0]` as a lexpr, so we can't allow it to be
        // coerced. But if the pattern matches by value, `f()[0]` is
        // still syntactically a lexpr, but we *do* want to allow
        // coercions.
        //
        // However, *likely* we are ok with allowing coercions to
        // happen if there are no explicit ref mut patterns - all
        // implicit ref mut patterns must occur behind a reference, so
        // they will have the "correct" variance and lifetime.
        //
        // This does mean that the following pattern would be legal:
        //
        // ```
        // struct Foo(Bar);
        // struct Bar(u32);
        // impl Deref for Foo {
        //     type Target = Bar;
        //     fn deref(&self) -> &Bar { &self.0 }
        // }
        // impl DerefMut for Foo {
        //     fn deref_mut(&mut self) -> &mut Bar { &mut self.0 }
        // }
        // fn foo(x: &mut Foo) {
        //     {
        //         let Bar(z): &mut Bar = x;
        //         *z = 42;
        //     }
        //     assert_eq!(foo.0.0, 42);
        // }
        // ```
        //
        // FIXME(tschottdorf): don't call contains_explicit_ref_binding, which
        // is problematic as the HIR is being scraped, but ref bindings may be
        // implicit after #42640. We need to make sure that pat_adjustments
        // (once introduced) is populated by the time we get here.
        //
        // See #44848.
        if contains_ref_bindings || no_arms {
            self.infer_expr_no_expect(scrut, scrutinee_is_read)
        } else {
            // ...but otherwise we want to use any supertype of the
            // scrutinee. This is sort of a workaround, see note (*) in
            // `check_pat` for some details.
            let scrut_ty = self.table.next_ty_var(scrut.into());
            self.infer_expr_coerce_never(scrut, &Expectation::HasType(scrut_ty), scrutinee_is_read);
            scrut_ty
        }
    }

    fn infer_expr_path(&mut self, path: &Path, id: ExprOrPatId, scope_id: ExprId) -> Ty<'db> {
        let g = self.resolver.update_to_inner_scope(self.db, self.store_owner, scope_id);
        let ty = match self.infer_path(path, id) {
            Some((_, ty)) => ty,
            None => {
                if path.mod_path().is_some_and(|mod_path| mod_path.is_ident() || mod_path.is_self())
                {
                    self.push_diagnostic(InferenceDiagnostic::UnresolvedIdent { id });
                }
                self.err_ty()
            }
        };
        self.resolver.reset_to_guard(g);
        ty
    }

    fn infer_unop_expr(
        &mut self,
        unop: UnaryOp,
        oprnd: ExprId,
        expected: &Expectation<'db>,
        expr: ExprId,
    ) -> Ty<'db> {
        let expected_inner = match unop {
            UnaryOp::Not | UnaryOp::Neg => expected,
            UnaryOp::Deref => &Expectation::None,
        };
        let mut oprnd_t = self.infer_expr_inner(oprnd, expected_inner, ExprIsRead::Yes);

        oprnd_t = self.structurally_resolve_type(oprnd.into(), oprnd_t);
        match unop {
            UnaryOp::Deref => {
                if let Some(ty) = self.lookup_derefing(expr, oprnd, oprnd_t) {
                    oprnd_t = ty;
                } else {
                    self.push_diagnostic(InferenceDiagnostic::CannotBeDereferenced {
                        expr,
                        found: oprnd_t.store(),
                    });
                    oprnd_t = self.types.types.error;
                }
            }
            UnaryOp::Not => {
                let result = self.infer_user_unop(expr, oprnd_t, unop);
                // If it's builtin, we can reuse the type, this helps inference.
                if !(oprnd_t.is_integral() || oprnd_t.kind() == TyKind::Bool) {
                    oprnd_t = result;
                }
            }
            UnaryOp::Neg => {
                let result = self.infer_user_unop(expr, oprnd_t, unop);
                // If it's builtin, we can reuse the type, this helps inference.
                if !oprnd_t.is_numeric() {
                    oprnd_t = result;
                }
            }
        }
        oprnd_t
    }

    fn infer_array_repeat_expr(
        &mut self,
        element: ExprId,
        count: ExprId,
        expected: &Expectation<'db>,
        expr: ExprId,
    ) -> Ty<'db> {
        let interner = self.interner();
        let count_ct = self.create_body_anon_const(count, self.types.types.usize, true);
        let count = self.table.try_structurally_resolve_const(count.into(), count_ct);

        let uty = match expected {
            Expectation::HasType(uty) => uty.builtin_index(),
            _ => None,
        };

        let t = match uty {
            Some(uty) => {
                self.infer_expr_coerce(element, &Expectation::has_type(uty), ExprIsRead::Yes);
                uty
            }
            None => {
                let ty = self.table.next_ty_var(element.into());
                self.infer_expr(element, &Expectation::has_type(ty), ExprIsRead::Yes);
                ty
            }
        };

        // We defer checking whether the element type is `Copy` as it is possible to have
        // an inference variable as a repeat count and it seems unlikely that `Copy` would
        // have inference side effects required for type checking to succeed.
        // FIXME: Do it here like rustc.
        // self.deferred_repeat_expr_checks.borrow_mut().push((element, element_ty, count));

        let ty = Ty::new_array_with_const_len(interner, t, count);
        self.table.register_wf_obligation(ty.into(), ObligationCause::new(expr));
        ty
    }

    fn infer_array_elements_expr(
        &mut self,
        args: &[ExprId],
        expected: &Expectation<'db>,
        expr: ExprId,
    ) -> Ty<'db> {
        let element_ty = if !args.is_empty() {
            let coerce_to = expected
                .to_option(&self.table)
                .and_then(|uty| {
                    self.table
                        .resolve_vars_with_obligations(uty)
                        .builtin_index()
                        // Avoid using the original type variable as the coerce_to type, as it may resolve
                        // during the first coercion instead of being the LUB type.
                        .filter(|t| !self.table.resolve_vars_with_obligations(*t).is_ty_var())
                })
                .unwrap_or_else(|| self.table.next_ty_var(expr.into()));
            let mut coerce = CoerceMany::with_coercion_sites(coerce_to, args);

            for &e in args {
                // FIXME: the element expectation should use
                // `try_structurally_resolve_and_adjust_for_branches` just like in `if` and `match`.
                // While that fixes nested coercion, it will break [some
                // code like this](https://github.com/rust-lang/rust/pull/140283#issuecomment-2958776528).
                // If we find a way to support recursive tuple coercion, this break can be avoided.
                let e_ty =
                    self.infer_expr_inner(e, &Expectation::has_type(coerce_to), ExprIsRead::Yes);
                let cause = ObligationCause::new(e);
                coerce.coerce(self, &cause, e, e_ty, ExprIsRead::Yes);
            }
            coerce.complete(self)
        } else {
            self.table.next_ty_var(expr.into())
        };
        let array_len = args.len() as u128;
        Ty::new_array(self.interner(), element_ty, array_len)
    }

    pub(super) fn infer_return(&mut self, expr: ExprId) {
        let ret_ty = self
            .return_coercion
            .as_mut()
            .expect("infer_return called outside function body")
            .expected_ty();
        let return_expr_ty =
            self.infer_expr_inner(expr, &Expectation::HasType(ret_ty), ExprIsRead::Yes);
        let mut coerce_many = self.return_coercion.take().unwrap();
        coerce_many.coerce(
            self,
            &ObligationCause::new(expr),
            expr,
            return_expr_ty,
            ExprIsRead::Yes,
        );
        self.return_coercion = Some(coerce_many);
    }

    fn infer_expr_return(&mut self, ret: ExprId, expr: Option<ExprId>) -> Ty<'db> {
        match self.return_coercion {
            Some(_) => {
                if let Some(expr) = expr {
                    self.infer_return(expr);
                } else {
                    let mut coerce = self.return_coercion.take().unwrap();
                    coerce.coerce_forced_unit(
                        self,
                        ret,
                        &ObligationCause::new(ret),
                        true,
                        ExprIsRead::Yes,
                    );
                    self.return_coercion = Some(coerce);
                }
            }
            None => {
                // FIXME: diagnose return outside of function
                if let Some(expr) = expr {
                    self.infer_expr_no_expect(expr, ExprIsRead::Yes);
                }
            }
        }
        self.types.types.never
    }

    fn infer_expr_become(&mut self, expr: ExprId) -> Ty<'db> {
        match &self.return_coercion {
            Some(return_coercion) => {
                let ret_ty = return_coercion.expected_ty();

                let call_expr_ty =
                    self.infer_expr_inner(expr, &Expectation::HasType(ret_ty), ExprIsRead::Yes);

                // NB: this should *not* coerce.
                //     tail calls don't support any coercions except lifetimes ones (like `&'static u8 -> &'a u8`).
                _ = self.demand_eqtype(expr.into(), call_expr_ty, ret_ty);
            }
            None => {
                // FIXME: diagnose `become` outside of functions
                self.infer_expr_no_expect(expr, ExprIsRead::Yes);
            }
        }

        self.types.types.never
    }

    fn infer_expr_box(&mut self, inner_expr: ExprId, expected: &Expectation<'db>) -> Ty<'db> {
        if let Some(box_id) = self.resolve_boxed_box() {
            let table = &mut self.table;
            let inner_exp = expected
                .to_option(table)
                .as_ref()
                .and_then(|e| e.as_adt())
                .filter(|(e_adt, _)| e_adt == &box_id)
                .map(|(_, subts)| {
                    let g = subts.type_at(0);
                    Expectation::rvalue_hint(self, g)
                })
                .unwrap_or_else(Expectation::none);

            let inner_ty = self.infer_expr_inner(inner_expr, &inner_exp, ExprIsRead::Yes);
            Ty::new_box(self.interner(), inner_ty)
        } else {
            self.err_ty()
        }
    }

    fn infer_block(
        &mut self,
        expr: ExprId,
        statements: &[Statement],
        tail: Option<ExprId>,
        label: Option<LabelId>,
        expected: &Expectation<'db>,
    ) -> Ty<'db> {
        let coerce_ty = expected.coercion_target_type(&mut self.table, expr.into());
        let g = self.resolver.update_to_inner_scope(self.db, self.store_owner, expr);

        let (break_ty, ty) =
            self.with_breakable_ctx(BreakableKind::Block, Some(coerce_ty), label, |this| {
                for stmt in statements {
                    match stmt {
                        Statement::Let { pat, type_ref, initializer, else_branch } => {
                            let decl_ty = type_ref
                                .as_ref()
                                .map(|&tr| this.make_body_ty(tr))
                                .unwrap_or_else(|| this.table.next_ty_var((*pat).into()));

                            let ty = if let Some(expr) = initializer {
                                // If we have a subpattern that performs a read, we want to consider this
                                // to diverge for compatibility to support something like `let x: () = *never_ptr;`.
                                let target_is_read =
                                    if this.pat_guaranteed_to_constitute_read_for_never(*pat) {
                                        ExprIsRead::Yes
                                    } else {
                                        ExprIsRead::No
                                    };
                                let ty = if this.contains_explicit_ref_binding(*pat) {
                                    this.infer_expr(
                                        *expr,
                                        &Expectation::has_type(decl_ty),
                                        target_is_read,
                                    )
                                } else {
                                    this.infer_expr_coerce(
                                        *expr,
                                        &Expectation::has_type(decl_ty),
                                        target_is_read,
                                    )
                                };
                                if type_ref.is_some() { decl_ty } else { ty }
                            } else {
                                decl_ty
                            };

                            this.infer_top_pat(
                                *pat,
                                ty,
                                PatOrigin::LetStmt { has_else: else_branch.is_some() },
                            );
                            if let Some(expr) = else_branch {
                                let previous_diverges =
                                    mem::replace(&mut this.diverges, Diverges::Maybe);
                                this.infer_expr_coerce(
                                    *expr,
                                    &Expectation::HasType(this.types.types.never),
                                    ExprIsRead::Yes,
                                );
                                this.diverges = previous_diverges;
                            }
                        }
                        &Statement::Expr { expr, has_semi } => {
                            if has_semi {
                                this.infer_expr(expr, &Expectation::none(), ExprIsRead::Yes);
                            } else {
                                this.infer_expr_coerce(
                                    expr,
                                    &Expectation::HasType(this.types.types.unit),
                                    ExprIsRead::Yes,
                                );
                            }
                        }
                        Statement::Item(_) => (),
                    }
                }

                // FIXME: This should make use of the breakable CoerceMany
                if let Some(expr) = tail {
                    this.infer_expr_coerce(expr, expected, ExprIsRead::Yes)
                } else {
                    // Citing rustc: if there is no explicit tail expression,
                    // that is typically equivalent to a tail expression
                    // of `()` -- except if the block diverges. In that
                    // case, there is no value supplied from the tail
                    // expression (assuming there are no other breaks,
                    // this implies that the type of the block will be
                    // `!`).
                    if this.diverges.is_always() {
                        // we don't even make an attempt at coercion
                        this.table.new_maybe_never_var(expr.into())
                    } else if let Some(t) = expected.only_has_type(&mut this.table) {
                        if this
                            .coerce(
                                expr,
                                this.types.types.unit,
                                t,
                                AllowTwoPhase::No,
                                ExprIsRead::Yes,
                            )
                            .is_err()
                        {
                            this.emit_type_mismatch(expr.into(), t, this.types.types.unit);
                        }
                        t
                    } else {
                        this.types.types.unit
                    }
                }
            });
        self.resolver.reset_to_guard(g);

        break_ty.unwrap_or(ty)
    }

    fn lookup_field(
        &mut self,
        field_expr: ExprId,
        receiver_ty: Ty<'db>,
        name: &Name,
    ) -> Option<(Ty<'db>, Either<FieldId, TupleFieldId>, Vec<Adjustment>, bool)> {
        let interner = self.interner();
        let mut autoderef = self.table.autoderef_with_tracking(receiver_ty, field_expr.into());
        let mut private_field = None;
        let res = autoderef.by_ref().find_map(|(derefed_ty, _)| {
            let (field_id, parameters) = match derefed_ty.kind() {
                TyKind::Tuple(substs) => {
                    return name.as_tuple_index().and_then(|idx| {
                        substs.as_slice().get(idx).copied().map(|ty| {
                            (
                                Either::Right(TupleFieldId {
                                    tuple: TupleId(
                                        self.tuple_field_accesses_rev.insert_full(substs).0 as u32,
                                    ),
                                    index: idx as u32,
                                }),
                                ty,
                            )
                        })
                    });
                }
                TyKind::Adt(adt, parameters) => match adt.def_id() {
                    hir_def::AdtId::StructId(s) => {
                        let local_id = s.fields(self.db).field(name)?;
                        let field = FieldId { parent: s.into(), local_id };
                        (field, parameters)
                    }
                    hir_def::AdtId::UnionId(u) => {
                        let local_id = u.fields(self.db).field(name)?;
                        let field = FieldId { parent: u.into(), local_id };
                        (field, parameters)
                    }
                    hir_def::AdtId::EnumId(_) => return None,
                },
                _ => return None,
            };
            let is_visible = VariantFields::field_visibilities(self.db, field_id.parent)
                [field_id.local_id]
                .is_visible_from(self.db, self.resolver.module());
            if !is_visible {
                if private_field.is_none() {
                    private_field = Some((field_id, parameters));
                }
                return None;
            }
            let ty = self.db.field_types(field_id.parent)[field_id.local_id]
                .get()
                .instantiate(interner, parameters)
                .skip_norm_wip();
            Some((Either::Left(field_id), ty))
        });

        Some(match res {
            Some((field_id, ty)) => {
                let adjustments =
                    self.table.register_infer_ok(autoderef.adjust_steps_as_infer_ok());
                let ty = self.process_remote_user_written_ty(ty);

                (ty, field_id, adjustments, true)
            }
            None => {
                let (field_id, subst) = private_field?;
                let adjustments =
                    self.table.register_infer_ok(autoderef.adjust_steps_as_infer_ok());
                let ty = self.db.field_types(field_id.parent)[field_id.local_id]
                    .get()
                    .instantiate(self.interner(), subst)
                    .skip_norm_wip();
                let ty = self.process_remote_user_written_ty(ty);

                (ty, Either::Left(field_id), adjustments, false)
            }
        })
    }

    fn infer_field_access(
        &mut self,
        tgt_expr: ExprId,
        receiver: ExprId,
        name: &Name,
        expected: &Expectation<'db>,
    ) -> Ty<'db> {
        // Field projections don't constitute reads.
        let receiver_ty = self.infer_expr_inner(receiver, &Expectation::none(), ExprIsRead::No);
        let receiver_ty = self.structurally_resolve_type(receiver.into(), receiver_ty);

        if name.is_missing() {
            // Bail out early, don't even try to look up field. Also, we don't issue an unresolved
            // field diagnostic because this is a syntax error rather than a semantic error.
            return self.err_ty();
        }

        match self.lookup_field(tgt_expr, receiver_ty, name) {
            Some((ty, field_id, adjustments, is_public)) => {
                self.write_expr_adj(receiver, adjustments.into_boxed_slice());
                self.result.field_resolutions.insert(tgt_expr, field_id);
                if !is_public && let Either::Left(field) = field_id {
                    // FIXME: Merge this diagnostic into UnresolvedField?
                    self.push_diagnostic(InferenceDiagnostic::PrivateField {
                        expr: tgt_expr,
                        field,
                    });
                }
                ty
            }
            None => {
                // no field found, lets attempt to resolve it like a function so that IDE things
                // work out while people are typing
                let resolved = self.lookup_method_including_private(
                    receiver_ty,
                    name.clone(),
                    None,
                    receiver,
                    tgt_expr,
                );
                self.push_diagnostic(InferenceDiagnostic::UnresolvedField {
                    expr: tgt_expr,
                    receiver: receiver_ty.store(),
                    name: name.clone(),
                    method_with_same_name_exists: resolved.is_ok(),
                });
                match resolved {
                    Ok((func, _is_visible)) => {
                        self.check_method_call(tgt_expr, &[], func.sig, expected)
                    }
                    Err(_) => self.err_ty(),
                }
            }
        }
    }

    fn instantiate_erroneous_method(&mut self, def_id: FunctionId) -> MethodCallee<'db> {
        // FIXME: Using fresh infer vars for the method args isn't optimal,
        // we can do better by going thorough the full probe/confirm machinery.
        let args = self.table.fresh_args_for_item(Span::Dummy, def_id.into());
        let sig = self
            .db
            .callable_item_signature(def_id.into())
            .instantiate(self.interner(), args)
            .skip_norm_wip();
        let sig = self.infcx().instantiate_binder_with_fresh_vars(
            Span::Dummy,
            BoundRegionConversionTime::FnCall,
            sig,
        );
        MethodCallee { def_id, args, sig }
    }

    fn infer_method_call_as_call(
        &mut self,
        tgt_expr: ExprId,
        args: &[ExprId],
        callee_ty: Ty<'db>,
        param_tys: &[Ty<'db>],
        ret_ty: Ty<'db>,
        indices_to_skip: &[u32],
        is_varargs: bool,
        expected: &Expectation<'db>,
    ) -> Ty<'db> {
        if let TyKind::FnDef(def_id, args) = callee_ty.kind() {
            let def_id = match def_id.0 {
                CallableDefId::FunctionId(it) => it.into(),
                CallableDefId::StructId(it) => it.into(),
                CallableDefId::EnumVariantId(it) => it.loc(self.db).parent.into(),
            };
            self.add_required_obligations_for_value_path(tgt_expr.into(), def_id, args);
        }

        self.check_call_arguments(
            tgt_expr,
            param_tys,
            ret_ty,
            expected,
            args,
            indices_to_skip,
            is_varargs,
            TupleArgumentsFlag::DontTupleArguments,
        );
        ret_ty
    }

    fn infer_method_call(
        &mut self,
        tgt_expr: ExprId,
        receiver: ExprId,
        args: &[ExprId],
        method_name: &Name,
        generic_args: Option<&HirGenericArgs>,
        expected: &Expectation<'db>,
    ) -> Ty<'db> {
        let receiver_ty = self.infer_expr_inner(receiver, &Expectation::none(), ExprIsRead::Yes);
        let receiver_ty = self.table.try_structurally_resolve_type(receiver.into(), receiver_ty);

        let resolved = self.lookup_method_including_private(
            receiver_ty,
            method_name.clone(),
            generic_args,
            receiver,
            tgt_expr,
        );
        match resolved {
            Ok((func, visible)) => {
                if !visible {
                    self.push_diagnostic(InferenceDiagnostic::PrivateAssocItem {
                        id: tgt_expr.into(),
                        item: func.def_id.into(),
                    })
                }
                self.check_method_call(tgt_expr, args, func.sig, expected)
            }
            // Failed to resolve, report diagnostic and try to resolve as call to field access or
            // assoc function
            Err(_) => {
                let field_with_same_name_exists =
                    match self.lookup_field(tgt_expr, receiver_ty, method_name) {
                        Some((ty, field_id, adjustments, _public)) => {
                            self.write_expr_adj(receiver, adjustments.into_boxed_slice());
                            self.result.field_resolutions.insert(tgt_expr, field_id);
                            Some(ty)
                        }
                        None => None,
                    };

                let assoc_func_with_same_name =
                    self.with_method_resolution(tgt_expr.into(), receiver.into(), |ctx| {
                        if !matches!(
                            receiver_ty.kind(),
                            TyKind::Infer(InferTy::TyVar(_)) | TyKind::Error(_)
                        ) {
                            ctx.probe_for_name(
                                method_resolution::Mode::Path,
                                method_name.clone(),
                                receiver_ty,
                            )
                        } else {
                            Err(MethodError::ErrorReported)
                        }
                    });
                let assoc_func_with_same_name = match assoc_func_with_same_name {
                    Ok(method_resolution::Pick {
                        item: CandidateId::FunctionId(def_id), ..
                    })
                    | Err(MethodError::PrivateMatch(method_resolution::Pick {
                        item: CandidateId::FunctionId(def_id),
                        ..
                    })) => Some(self.instantiate_erroneous_method(def_id)),
                    _ => None,
                };

                self.push_diagnostic(InferenceDiagnostic::UnresolvedMethodCall {
                    expr: tgt_expr,
                    receiver: receiver_ty.store(),
                    name: method_name.clone(),
                    field_with_same_name: field_with_same_name_exists.map(|it| it.store()),
                    assoc_func_with_same_name: assoc_func_with_same_name.map(|it| it.def_id),
                });

                let recovered = match assoc_func_with_same_name {
                    Some(it) => Some((
                        Ty::new_fn_def(
                            self.interner(),
                            CallableDefId::FunctionId(it.def_id).into(),
                            it.args,
                        ),
                        it.sig,
                        true,
                    )),
                    None => field_with_same_name_exists.and_then(|field_ty| {
                        let callable_sig = field_ty.callable_sig(self.interner())?;
                        Some((field_ty, callable_sig.skip_binder(), false))
                    }),
                };
                match recovered {
                    Some((callee_ty, sig, strip_first)) => self.infer_method_call_as_call(
                        tgt_expr,
                        args,
                        callee_ty,
                        sig.inputs_and_output.inputs().get(strip_first as usize..).unwrap_or(&[]),
                        sig.output(),
                        &[],
                        true,
                        expected,
                    ),
                    None => {
                        for &arg in args.iter() {
                            self.infer_expr_no_expect(arg, ExprIsRead::Yes);
                        }
                        self.err_ty()
                    }
                }
            }
        }
    }

    fn check_method_call(
        &mut self,
        tgt_expr: ExprId,
        args: &[ExprId],
        sig: FnSig<'db>,
        expected: &Expectation<'db>,
    ) -> Ty<'db> {
        let param_tys = if !sig.inputs_and_output.inputs().is_empty() {
            &sig.inputs_and_output.inputs()[1..]
        } else {
            &[]
        };
        let ret_ty = sig.output();

        self.check_call_arguments(
            tgt_expr,
            param_tys,
            ret_ty,
            expected,
            args,
            &[],
            sig.c_variadic(),
            TupleArgumentsFlag::DontTupleArguments,
        );
        ret_ty
    }

    /// Generic function that factors out common logic from function calls,
    /// method calls and overloaded operators.
    pub(super) fn check_call_arguments(
        &mut self,
        call_expr: ExprId,
        // Types (as defined in the *signature* of the target function)
        formal_input_tys: &[Ty<'db>],
        formal_output: Ty<'db>,
        // Expected output from the parent expression or statement
        expectation: &Expectation<'db>,
        // The expressions for each provided argument
        provided_args: &[ExprId],
        skip_indices: &[u32],
        // Whether the function is variadic, for example when imported from C
        c_variadic: bool,
        // Whether the arguments have been bundled in a tuple (ex: closures)
        tuple_arguments: TupleArgumentsFlag,
    ) {
        let formal_input_tys: Vec<_> = formal_input_tys
            .iter()
            .map(|&ty| {
                let generalized_ty = self.table.next_ty_var(call_expr.into());
                let _ = self.demand_eqtype(call_expr.into(), ty, generalized_ty);
                generalized_ty
            })
            .collect();

        // First, let's unify the formal method signature with the expectation eagerly.
        // We use this to guide coercion inference; it's output is "fudged" which means
        // any remaining type variables are assigned to new, unrelated variables. This
        // is because the inference guidance here is only speculative.
        let formal_output = self.table.resolve_vars_with_obligations(formal_output);
        let expected_input_tys: Option<Vec<_>> = expectation
            .only_has_type(&mut self.table)
            .and_then(|expected_output| {
                self.table
                    .infer_ctxt
                    .fudge_inference_if_ok(|| {
                        let mut ocx = ObligationCtxt::new(&self.table.infer_ctxt);

                        // Attempt to apply a subtyping relationship between the formal
                        // return type (likely containing type variables if the function
                        // is polymorphic) and the expected return type.
                        // No argument expectations are produced if unification fails.
                        let origin = ObligationCause::new(call_expr);
                        ocx.sup(&origin, self.table.param_env, expected_output, formal_output)?;

                        for &ty in &formal_input_tys {
                            ocx.register_obligation(Obligation::new(
                                self.interner(),
                                ObligationCause::new(call_expr),
                                self.table.param_env,
                                ClauseKind::WellFormed(ty.into()),
                            ));
                        }

                        if !ocx.try_evaluate_obligations().is_empty() {
                            return Err(TypeError::Mismatch);
                        }

                        // Record all the argument types, with the args
                        // produced from the above subtyping unification.
                        Ok(Some(formal_input_tys.clone()))
                    })
                    .ok()
            })
            .unwrap_or_default();

        // If the arguments should be wrapped in a tuple (ex: closures), unwrap them here
        let (formal_input_tys, expected_input_tys) = if tuple_arguments
            == TupleArgumentsFlag::TupleArguments
        {
            let tuple_type = self.structurally_resolve_type(call_expr.into(), formal_input_tys[0]);
            match tuple_type.kind() {
                // We expected a tuple and got a tuple
                TyKind::Tuple(arg_types) => {
                    // Argument length differs
                    if arg_types.len() != provided_args.len() {
                        // FIXME: Emit an error.
                    }
                    let expected_input_tys = match expected_input_tys {
                        Some(expected_input_tys) => match expected_input_tys.first() {
                            Some(ty) => match ty.kind() {
                                TyKind::Tuple(tys) => Some(tys.iter().collect()),
                                _ => None,
                            },
                            None => None,
                        },
                        None => None,
                    };
                    (arg_types.to_vec(), expected_input_tys)
                }
                _ => {
                    // Otherwise, there's a mismatch, so clear out what we're expecting, and set
                    // our input types to err_args so we don't blow up the error messages
                    // FIXME: Emit an error.
                    (vec![self.types.types.error; provided_args.len()], None)
                }
            }
        } else {
            (formal_input_tys, expected_input_tys)
        };

        // If there are no external expectations at the call site, just use the types from the function defn
        let expected_input_tys = if let Some(expected_input_tys) = expected_input_tys {
            assert_eq!(expected_input_tys.len(), formal_input_tys.len());
            expected_input_tys
        } else {
            formal_input_tys.clone()
        };

        let minimum_input_count = expected_input_tys.len();
        let provided_arg_count = provided_args.len() - skip_indices.len();

        // Keep track of whether we *could possibly* be satisfied, i.e. whether we're on the happy path
        // if the wrong number of arguments were supplied, we CAN'T be satisfied,
        // and if we're c_variadic, the supplied arguments must be >= the minimum count from the function
        // otherwise, they need to be identical, because rust doesn't currently support variadic functions
        let args_count_matches = if c_variadic {
            provided_arg_count >= minimum_input_count
        } else {
            provided_arg_count == minimum_input_count
        };

        if !args_count_matches {
            self.push_diagnostic(InferenceDiagnostic::MismatchedArgCount {
                call_expr,
                expected: expected_input_tys.len() + skip_indices.len(),
                found: provided_args.len(),
            });
        }

        // We introduce a helper function to demand that a given argument satisfy a given input
        // This is more complicated than just checking type equality, as arguments could be coerced
        // This version writes those types back so further type checking uses the narrowed types
        let demand_compatible = |this: &mut InferenceContext<'_, 'db>, idx| {
            let formal_input_ty: Ty<'db> = formal_input_tys[idx];
            let expected_input_ty: Ty<'db> = expected_input_tys[idx];
            let provided_arg = provided_args[idx];

            debug!("checking argument {}: {:?} = {:?}", idx, provided_arg, formal_input_ty);

            // We're on the happy path here, so we'll do a more involved check and write back types
            // To check compatibility, we'll do 3 things:
            // 1. Unify the provided argument with the expected type
            let expectation = Expectation::rvalue_hint(this, expected_input_ty);

            let checked_ty = this.infer_expr_inner(provided_arg, &expectation, ExprIsRead::Yes);

            // 2. Coerce to the most detailed type that could be coerced
            //    to, which is `expected_ty` if `rvalue_hint` returns an
            //    `ExpectHasType(expected_ty)`, or the `formal_ty` otherwise.
            let coerced_ty = expectation.only_has_type(&mut this.table).unwrap_or(formal_input_ty);

            // Cause selection errors caused by resolving a single argument to point at the
            // argument and not the call. This lets us customize the span pointed to in the
            // fulfillment error to be more accurate.
            let coerced_ty = this.table.resolve_vars_with_obligations(coerced_ty);

            let coerce_error = this
                .coerce(provided_arg, checked_ty, coerced_ty, AllowTwoPhase::Yes, ExprIsRead::Yes)
                .err();
            if coerce_error.is_some() {
                return Err((coerce_error, coerced_ty, checked_ty));
            }

            // 3. Check if the formal type is actually equal to the checked one
            //    and register any such obligations for future type checks.
            let formal_ty_error = this
                .table
                .infer_ctxt
                .at(&ObligationCause::new(provided_arg), this.table.param_env)
                .eq(formal_input_ty, coerced_ty);

            // If neither check failed, the types are compatible
            match formal_ty_error {
                Ok(InferOk { obligations, value: () }) => {
                    this.table.register_predicates(obligations);
                    Ok(())
                }
                Err(err) => Err((Some(err), coerced_ty, checked_ty)),
            }
        };

        // Check the arguments.
        // We do this in a pretty awful way: first we type-check any arguments
        // that are not closures, then we type-check the closures. This is so
        // that we have more information about the types of arguments when we
        // type-check the functions. This isn't really the right way to do this.
        for check_closures in [false, true] {
            // More awful hacks: before we check argument types, try to do
            // an "opportunistic" trait resolution of any trait bounds on
            // the call. This helps coercions.
            if check_closures {
                self.table.select_obligations_where_possible();
            }

            let mut skip_indices = skip_indices.iter().copied();
            // Check each argument, to satisfy the input it was provided for
            // Visually, we're traveling down the diagonal of the compatibility matrix
            for (idx, arg) in provided_args.iter().enumerate() {
                if skip_indices.clone().next() == Some(idx as u32) {
                    skip_indices.next();
                    continue;
                }

                // For this check, we do *not* want to treat async coroutine closures (async blocks)
                // as proper closures. Doing so would regress type inference when feeding
                // the return value of an argument-position async block to an argument-position
                // closure wrapped in a block.
                // See <https://github.com/rust-lang/rust/issues/112225>.
                let is_closure = if let Expr::Closure { closure_kind, .. } = self.store[*arg] {
                    !matches!(closure_kind, ClosureKind::OldCoroutine(_))
                } else {
                    false
                };
                if is_closure != check_closures {
                    continue;
                }

                if idx >= minimum_input_count {
                    // Make sure we've checked this expr at least once.
                    self.infer_expr_no_expect(*arg, ExprIsRead::Yes);
                    continue;
                }

                if let Err((_error, expected, found)) = demand_compatible(self, idx)
                    && args_count_matches
                {
                    // Don't report type mismatches if there is a mismatch in args count.
                    self.emit_type_mismatch((*arg).into(), expected, found);
                }
            }
        }

        if !args_count_matches {}
    }

    pub(super) fn with_breakable_ctx<T>(
        &mut self,
        kind: BreakableKind,
        ty: Option<Ty<'db>>,
        label: Option<LabelId>,
        cb: impl FnOnce(&mut Self) -> T,
    ) -> (Option<Ty<'db>>, T) {
        self.breakables.push({
            BreakableContext { kind, may_break: false, coerce: ty.map(CoerceMany::new), label }
        });
        let res = cb(self);
        let ctx = self.breakables.pop().expect("breakable stack broken");
        (if ctx.may_break { ctx.coerce.map(|ctx| ctx.complete(self)) } else { None }, res)
    }
}

/// Controls whether the arguments are tupled. This is used for the call
/// operator.
///
/// Tupling means that all call-side arguments are packed into a tuple and
/// passed as a single parameter. For example, if tupling is enabled, this
/// function:
/// ```
/// fn f(x: (isize, isize)) {}
/// ```
/// Can be called as:
/// ```ignore UNSOLVED (can this be done in user code?)
/// # fn f(x: (isize, isize)) {}
/// f(1, 2);
/// ```
/// Instead of:
/// ```
/// # fn f(x: (isize, isize)) {}
/// f((1, 2));
/// ```
#[derive(Copy, Clone, Eq, PartialEq)]
pub(super) enum TupleArgumentsFlag {
    DontTupleArguments,
    TupleArguments,
}
