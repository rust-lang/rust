//! Type inference for expressions.

use std::{iter::repeat_with, mem};

use either::Either;
use hir_def::{
    FieldId, GenericDefId, ItemContainerId, Lookup, TupleFieldId, TupleId,
    expr_store::path::{GenericArgs as HirGenericArgs, Path},
    hir::{
        Array, AsmOperand, AsmOptions, BinaryOp, BindingAnnotation, Expr, ExprId, ExprOrPatId,
        InlineAsmKind, LabelId, Literal, Pat, PatId, RecordSpread, Statement, UnaryOp,
    },
    resolver::ValueNs,
};
use hir_def::{FunctionId, hir::ClosureKind};
use hir_expand::name::Name;
use rustc_ast_ir::Mutability;
use rustc_type_ir::{
    CoroutineArgs, CoroutineArgsParts, InferTy, Interner,
    inherent::{AdtDef, GenericArgs as _, IntoKind, Ty as _},
};
use syntax::ast::RangeOp;
use tracing::debug;

use crate::{
    Adjust, Adjustment, CallableDefId, DeclContext, DeclOrigin, Rawness,
    autoderef::InferenceContextAutoderef,
    consteval,
    db::InternedCoroutine,
    generics::generics,
    infer::{
        AllowTwoPhase, BreakableKind, coerce::CoerceMany, find_continuable,
        pat::contains_explicit_ref_binding,
    },
    lower::{GenericPredicates, lower_mutability},
    method_resolution::{self, CandidateId, MethodCallee, MethodError},
    next_solver::{
        ErrorGuaranteed, FnSig, GenericArg, GenericArgs, TraitRef, Ty, TyKind, TypeError,
        infer::{
            BoundRegionConversionTime, InferOk,
            traits::{Obligation, ObligationCause},
        },
        obligation_ctxt::ObligationCtxt,
        util::clauses_as_obligations,
    },
    traits::FnTrait,
};

use super::{
    BreakableContext, Diverges, Expectation, InferenceContext, InferenceDiagnostic, TypeMismatch,
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
            let could_unify = self.unify(ty, expected_ty);
            if !could_unify {
                self.result.type_mismatches.get_or_insert_default().insert(
                    tgt_expr.into(),
                    TypeMismatch { expected: expected_ty.store(), actual: ty.store() },
                );
            }
        }
        ty
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
            match self.coerce(expr.into(), ty, target, AllowTwoPhase::No, is_read) {
                Ok(res) => res,
                Err(_) => {
                    self.result.type_mismatches.get_or_insert_default().insert(
                        expr.into(),
                        TypeMismatch { expected: target.store(), actual: ty.store() },
                    );
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
        match &self.body[pat] {
            // Does not constitute a read.
            Pat::Wild => false,

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
            | Pat::Ref { .. }
            | Pat::Lit(_)
            | Pat::Range { .. }
            | Pat::Slice { .. }
            | Pat::ConstBlock(_)
            | Pat::Record { .. }
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
        if let Pat::Bind { id, .. } = self.body[pat]
            && matches!(self.body[id].mode, BindingAnnotation::Ref | BindingAnnotation::RefMut)
        {
            return true;
        }

        let mut result = false;
        self.body.walk_pats_shallow(pat, |pat| result |= self.contains_explicit_ref_binding(pat));
        result
    }

    fn is_syntactic_place_expr(&self, expr: ExprId) -> bool {
        match &self.body[expr] {
            // Lang item paths cannot currently be local variables or statics.
            Expr::Path(Path::LangItem(_, _)) => false,
            Expr::Path(Path::Normal(path)) => path.type_anchor.is_none(),
            Expr::Path(path) => self
                .resolver
                .resolve_path_in_value_ns_fully(self.db, path, self.body.expr_path_hygiene(expr))
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
            | Expr::Async { .. }
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

    #[expect(clippy::needless_return)]
    pub(crate) fn check_lhs_assignable(&self, lhs: ExprId) {
        if self.is_syntactic_place_expr(lhs) {
            return;
        }

        // FIXME: Emit diagnostic.
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
                self.coerce(expr.into(), ty, target, AllowTwoPhase::No, ExprIsRead::Yes)
                    .expect("never-to-any coercion should always succeed")
            } else {
                ty
            }
        } else {
            if let Some(expected_ty) = expected.only_has_type(&mut self.table) {
                let could_unify = self.unify(ty, expected_ty);
                if !could_unify {
                    self.result.type_mismatches.get_or_insert_default().insert(
                        expr.into(),
                        TypeMismatch { expected: expected_ty.store(), actual: ty.store() },
                    );
                }
            }
            ty
        }
    }

    #[tracing::instrument(level = "debug", skip(self, is_read), ret)]
    fn infer_expr_inner(
        &mut self,
        tgt_expr: ExprId,
        expected: &Expectation<'db>,
        is_read: ExprIsRead,
    ) -> Ty<'db> {
        self.db.unwind_if_revision_cancelled();

        let expr = &self.body[tgt_expr];
        tracing::trace!(?expr);
        let ty = match expr {
            Expr::Missing => self.err_ty(),
            &Expr::If { condition, then_branch, else_branch } => {
                let expected = &expected.adjust_for_branches(&mut self.table);
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
                    expected.coercion_target_type(&mut self.table),
                    &coercion_sites,
                );
                coerce.coerce(self, &ObligationCause::new(), then_branch, then_ty, ExprIsRead::Yes);
                match else_branch {
                    Some(else_branch) => {
                        let else_ty = self.infer_expr_inner(else_branch, expected, ExprIsRead::Yes);
                        let else_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
                        coerce.coerce(
                            self,
                            &ObligationCause::new(),
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
                            &ObligationCause::new(),
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
                self.infer_top_pat(
                    pat,
                    input_ty,
                    Some(DeclContext { origin: DeclOrigin::LetExpr }),
                );
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
            Expr::Async { id: _, statements, tail } => {
                self.infer_async_block(tgt_expr, statements, tail)
            }
            &Expr::Loop { body, label } => {
                // FIXME: should be:
                // let ty = expected.coercion_target_type(&mut self.table);
                let ty = self.table.next_ty_var();
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
                        self.infer_top_pat(arm.pat, input_ty, None);
                    }

                    let expected = expected.adjust_for_branches(&mut self.table);
                    let result_ty = match &expected {
                        // We don't coerce to `()` so that if the match expression is a
                        // statement it's branches can have any consistent type.
                        Expectation::HasType(ty) if *ty != self.types.types.unit => *ty,
                        _ => self.table.next_ty_var(),
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
                            &ObligationCause::new(),
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
                            coerce.coerce(
                                self,
                                &ObligationCause::new(),
                                expr.unwrap_or(tgt_expr),
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
                            tgt_expr.into(),
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
                let (ty, def_id) = self.resolve_variant(tgt_expr.into(), path.as_deref(), false);

                if let Some(t) = expected.only_has_type(&mut self.table) {
                    self.unify(ty, t);
                }

                let substs = ty.as_adt().map(|(_, s)| s).unwrap_or(self.types.empty.generic_args);
                if let Some(variant) = def_id {
                    self.write_variant_resolution(tgt_expr.into(), variant);
                }
                match def_id {
                    _ if fields.is_empty() => {}
                    Some(def) => {
                        let field_types = self.db.field_types(def);
                        let variant_data = def.fields(self.db);
                        let visibilities = self.db.field_visibilities(def);
                        for field in fields.iter() {
                            let field_def = {
                                match variant_data.field(&field.name) {
                                    Some(local_id) => {
                                        if !visibilities[local_id]
                                            .is_visible_from(self.db, self.resolver.module())
                                        {
                                            self.push_diagnostic(
                                                InferenceDiagnostic::NoSuchField {
                                                    field: field.expr.into(),
                                                    private: Some(local_id),
                                                    variant: def,
                                                },
                                            );
                                        }
                                        Some(local_id)
                                    }
                                    None => {
                                        self.push_diagnostic(InferenceDiagnostic::NoSuchField {
                                            field: field.expr.into(),
                                            private: None,
                                            variant: def,
                                        });
                                        None
                                    }
                                }
                            };
                            let field_ty = field_def.map_or(self.err_ty(), |it| {
                                field_types[it].get().instantiate(self.interner(), &substs)
                            });

                            // Field type might have some unknown types
                            // FIXME: we may want to emit a single type variable for all instance of type fields?
                            let field_ty = self.insert_type_vars(field_ty);
                            self.infer_expr_coerce(
                                field.expr,
                                &Expectation::has_type(field_ty),
                                ExprIsRead::Yes,
                            );
                        }
                    }
                    None => {
                        for field in fields.iter() {
                            // Field projections don't constitute reads.
                            self.infer_expr_coerce(field.expr, &Expectation::None, ExprIsRead::No);
                        }
                    }
                }
                if let RecordSpread::Expr(expr) = *spread {
                    self.infer_expr(expr, &Expectation::has_type(ty), ExprIsRead::Yes);
                }
                ty
            }
            Expr::Field { expr, name } => self.infer_field_access(tgt_expr, *expr, name, expected),
            Expr::Await { expr } => {
                let inner_ty = self.infer_expr_inner(*expr, &Expectation::none(), ExprIsRead::Yes);
                self.resolve_associated_type(inner_ty, self.resolve_future_future_output())
            }
            Expr::Cast { expr, type_ref } => {
                let cast_ty = self.make_body_ty(*type_ref);
                let expr_ty =
                    self.infer_expr(*expr, &Expectation::Castable(cast_ty), ExprIsRead::Yes);
                self.deferred_cast_checks.push(CastCheck::new(tgt_expr, *expr, expr_ty, cast_ty));
                cast_ty
            }
            Expr::Ref { expr, rawness, mutability } => {
                let mutability = lower_mutability(*mutability);
                let expectation = if let Some((exp_inner, exp_rawness, exp_mutability)) = expected
                    .only_has_type(&mut self.table)
                    .as_ref()
                    .and_then(|t| t.as_reference_or_ptr())
                {
                    if exp_mutability == Mutability::Mut && mutability == Mutability::Not {
                        // FIXME: record type error - expected mut reference but found shared ref,
                        // which cannot be coerced
                    }
                    if exp_rawness == Rawness::Ref && *rawness == Rawness::RawPtr {
                        // FIXME: record type error - expected reference but found ptr,
                        // which cannot be coerced
                    }
                    Expectation::rvalue_hint(self, exp_inner)
                } else {
                    Expectation::none()
                };
                let inner_ty = self.infer_expr_inner(*expr, &expectation, ExprIsRead::Yes);
                match rawness {
                    Rawness::RawPtr => Ty::new_ptr(self.interner(), inner_ty, mutability),
                    Rawness::Ref => {
                        let lt = self.table.next_region_var();
                        Ty::new_ref(self.interner(), lt, inner_ty, mutability)
                    }
                }
            }
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
                let lhs_ty = match &self.body[target] {
                    // LHS of assignment doesn't constitute reads.
                    &Pat::Expr(expr) => {
                        Some(self.infer_expr(expr, &Expectation::none(), ExprIsRead::No))
                    }
                    Pat::Path(path) => {
                        let resolver_guard =
                            self.resolver.update_to_inner_scope(self.db, self.owner, tgt_expr);
                        let resolution = self.resolver.resolve_path_in_value_ns_fully(
                            self.db,
                            path,
                            self.body.pat_path_hygiene(target),
                        );
                        self.resolver.reset_to_guard(resolver_guard);

                        if matches!(
                            resolution,
                            Some(
                                ValueNs::ConstId(_)
                                    | ValueNs::StructId(_)
                                    | ValueNs::EnumVariantId(_)
                            )
                        ) {
                            None
                        } else {
                            Some(self.infer_expr_path(path, target.into(), tgt_expr))
                        }
                    }
                    _ => None,
                };
                let is_destructuring_assignment = lhs_ty.is_none();

                if let Some(lhs_ty) = lhs_ty {
                    self.write_pat_ty(target, lhs_ty);
                    self.infer_expr_coerce(value, &Expectation::has_type(lhs_ty), ExprIsRead::No);
                } else {
                    let rhs_ty = self.infer_expr(value, &Expectation::none(), ExprIsRead::Yes);
                    let resolver_guard =
                        self.resolver.update_to_inner_scope(self.db, self.owner, tgt_expr);
                    self.inside_assignment = true;
                    self.infer_top_pat(target, rhs_ty, None);
                    self.inside_assignment = false;
                    self.resolver.reset_to_guard(resolver_guard);
                }
                if is_destructuring_assignment && self.diverges.is_always() {
                    // Ordinary assignments always return `()`, even when they diverge.
                    // However, rustc lowers destructuring assignments into blocks, and blocks return `!` if they have no tail
                    // expression and they diverge. Therefore, we have to do the same here, even though we don't lower destructuring
                    // assignments into blocks.
                    self.table.new_maybe_never_var()
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

                let base_t = self.table.structurally_resolve_type(base_t);
                match self.lookup_indexing(tgt_expr, *base, base_t, idx_t) {
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
                    .map(|t| self.table.try_structurally_resolve_type(t).kind())
                {
                    Some(TyKind::Tuple(substs)) => substs
                        .iter()
                        .chain(repeat_with(|| self.table.next_ty_var()))
                        .take(exprs.len())
                        .collect::<Vec<_>>(),
                    _ => (0..exprs.len()).map(|_| self.table.next_ty_var()).collect(),
                };

                for (expr, ty) in exprs.iter().zip(tys.iter_mut()) {
                    *ty =
                        self.infer_expr_coerce(*expr, &Expectation::has_type(*ty), ExprIsRead::Yes);
                }

                Ty::new_tup(self.interner(), &tys)
            }
            Expr::Array(array) => self.infer_expr_array(array, expected),
            Expr::Literal(lit) => match lit {
                Literal::Bool(..) => self.types.types.bool,
                Literal::String(..) => self.types.types.static_str_ref,
                Literal::ByteString(bs) => {
                    let byte_type = self.types.types.u8;

                    let len = consteval::usize_const(
                        self.db,
                        Some(bs.len() as u128),
                        self.resolver.krate(),
                    );

                    let array_type = Ty::new_array_with_const_len(self.interner(), byte_type, len);
                    Ty::new_ref(
                        self.interner(),
                        self.types.regions.statik,
                        array_type,
                        Mutability::Not,
                    )
                }
                Literal::CString(..) => Ty::new_ref(
                    self.interner(),
                    self.types.regions.statik,
                    self.lang_items.CStr.map_or_else(
                        || self.err_ty(),
                        |strukt| {
                            Ty::new_adt(
                                self.interner(),
                                strukt.into(),
                                self.types.empty.generic_args,
                            )
                        },
                    ),
                    Mutability::Not,
                ),
                Literal::Char(..) => self.types.types.char,
                Literal::Int(_v, ty) => match ty {
                    Some(int_ty) => match int_ty {
                        hir_def::builtin_type::BuiltinInt::Isize => self.types.types.isize,
                        hir_def::builtin_type::BuiltinInt::I8 => self.types.types.i8,
                        hir_def::builtin_type::BuiltinInt::I16 => self.types.types.i16,
                        hir_def::builtin_type::BuiltinInt::I32 => self.types.types.i32,
                        hir_def::builtin_type::BuiltinInt::I64 => self.types.types.i64,
                        hir_def::builtin_type::BuiltinInt::I128 => self.types.types.i128,
                    },
                    None => {
                        let expected_ty = expected.to_option(&mut self.table);
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
                    }
                },
                Literal::Uint(_v, ty) => match ty {
                    Some(int_ty) => match int_ty {
                        hir_def::builtin_type::BuiltinUint::Usize => self.types.types.usize,
                        hir_def::builtin_type::BuiltinUint::U8 => self.types.types.u8,
                        hir_def::builtin_type::BuiltinUint::U16 => self.types.types.u16,
                        hir_def::builtin_type::BuiltinUint::U32 => self.types.types.u32,
                        hir_def::builtin_type::BuiltinUint::U64 => self.types.types.u64,
                        hir_def::builtin_type::BuiltinUint::U128 => self.types.types.u128,
                    },
                    None => {
                        let expected_ty = expected.to_option(&mut self.table);
                        let opt_ty = match expected_ty.as_ref().map(|it| it.kind()) {
                            Some(TyKind::Int(_) | TyKind::Uint(_)) => expected_ty,
                            Some(TyKind::Char) => Some(self.types.types.u8),
                            Some(TyKind::RawPtr(..) | TyKind::FnDef(..) | TyKind::FnPtr(..)) => {
                                Some(self.types.types.usize)
                            }
                            _ => None,
                        };
                        opt_ty.unwrap_or_else(|| self.table.next_int_var())
                    }
                },
                Literal::Float(_v, ty) => match ty {
                    Some(float_ty) => match float_ty {
                        hir_def::builtin_type::BuiltinFloat::F16 => self.types.types.f16,
                        hir_def::builtin_type::BuiltinFloat::F32 => self.types.types.f32,
                        hir_def::builtin_type::BuiltinFloat::F64 => self.types.types.f64,
                        hir_def::builtin_type::BuiltinFloat::F128 => self.types.types.f128,
                    },
                    None => {
                        let opt_ty = expected
                            .to_option(&mut self.table)
                            .filter(|ty| matches!(ty.kind(), TyKind::Float(_)));
                        opt_ty.unwrap_or_else(|| self.table.next_float_var())
                    }
                },
            },
            Expr::Underscore => {
                // Underscore expression is an error, we render a specialized diagnostic
                // to let the user know what type is expected though.
                let expected = expected.to_option(&mut self.table).unwrap_or_else(|| self.err_ty());
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
                        let ty = this.table.structurally_resolve_type(ty);
                        match ty.kind() {
                            TyKind::FnDef(def, parameters) => {
                                let fnptr_ty = Ty::new_fn_ptr(
                                    this.interner(),
                                    this.interner()
                                        .fn_sig(def)
                                        .instantiate(this.interner(), parameters),
                                );
                                _ = this.coerce(
                                    expr.into(),
                                    ty,
                                    fnptr_ty,
                                    AllowTwoPhase::No,
                                    ExprIsRead::Yes,
                                );
                            }
                            TyKind::Ref(_, base_ty, mutbl) => {
                                let ptr_ty = Ty::new_ptr(this.interner(), base_ty, mutbl);
                                _ = this.coerce(
                                    expr.into(),
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
        // use a new type variable if we got unknown here
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
            let scrut_ty = self.table.next_ty_var();
            self.infer_expr_coerce_never(scrut, &Expectation::HasType(scrut_ty), scrutinee_is_read);
            scrut_ty
        }
    }

    fn infer_expr_path(&mut self, path: &Path, id: ExprOrPatId, scope_id: ExprId) -> Ty<'db> {
        let g = self.resolver.update_to_inner_scope(self.db, self.owner, scope_id);
        let ty = match self.infer_path(path, id) {
            Some(ty) => ty,
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

        oprnd_t = self.table.structurally_resolve_type(oprnd_t);
        match unop {
            UnaryOp::Deref => {
                if let Some(ty) = self.lookup_derefing(expr, oprnd, oprnd_t) {
                    oprnd_t = ty;
                } else {
                    // FIXME: Report an error.
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

    fn infer_async_block(
        &mut self,
        tgt_expr: ExprId,
        statements: &[Statement],
        tail: &Option<ExprId>,
    ) -> Ty<'db> {
        let ret_ty = self.table.next_ty_var();
        let prev_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
        let prev_ret_ty = mem::replace(&mut self.return_ty, ret_ty);
        let prev_ret_coercion = self.return_coercion.replace(CoerceMany::new(ret_ty));

        // FIXME: We should handle async blocks like we handle closures
        let expected = &Expectation::has_type(ret_ty);
        let (_, inner_ty) = self.with_breakable_ctx(BreakableKind::Border, None, None, |this| {
            let ty = this.infer_block(tgt_expr, statements, *tail, None, expected);
            if let Some(target) = expected.only_has_type(&mut this.table) {
                match this.coerce(tgt_expr.into(), ty, target, AllowTwoPhase::No, ExprIsRead::Yes) {
                    Ok(res) => res,
                    Err(_) => {
                        this.result.type_mismatches.get_or_insert_default().insert(
                            tgt_expr.into(),
                            TypeMismatch { expected: target.store(), actual: ty.store() },
                        );
                        target
                    }
                }
            } else {
                ty
            }
        });

        self.diverges = prev_diverges;
        self.return_ty = prev_ret_ty;
        self.return_coercion = prev_ret_coercion;

        self.lower_async_block_type_impl_trait(inner_ty, tgt_expr)
    }

    pub(crate) fn lower_async_block_type_impl_trait(
        &mut self,
        inner_ty: Ty<'db>,
        tgt_expr: ExprId,
    ) -> Ty<'db> {
        let coroutine_id = InternedCoroutine(self.owner, tgt_expr);
        let coroutine_id = self.db.intern_coroutine(coroutine_id).into();
        let parent_args = GenericArgs::identity_for_item(self.interner(), self.generic_def.into());
        Ty::new_coroutine(
            self.interner(),
            coroutine_id,
            CoroutineArgs::new(
                self.interner(),
                CoroutineArgsParts {
                    parent_args: parent_args.as_slice(),
                    kind_ty: self.types.types.unit,
                    // rustc uses a special lang item type for the resume ty. I don't believe this can cause us problems.
                    resume_ty: self.types.types.unit,
                    yield_ty: self.types.types.unit,
                    return_ty: inner_ty,
                    // FIXME: Infer upvars.
                    tupled_upvars_ty: self.types.types.unit,
                },
            )
            .args,
        )
    }

    pub(crate) fn write_fn_trait_method_resolution(
        &mut self,
        fn_x: FnTrait,
        derefed_callee: Ty<'db>,
        adjustments: &mut Vec<Adjustment>,
        callee_ty: Ty<'db>,
        params: &[Ty<'db>],
        tgt_expr: ExprId,
    ) {
        match fn_x {
            FnTrait::FnOnce | FnTrait::AsyncFnOnce => (),
            FnTrait::FnMut | FnTrait::AsyncFnMut => {
                if let TyKind::Ref(lt, inner, Mutability::Mut) = derefed_callee.kind() {
                    if adjustments
                        .last()
                        .map(|it| matches!(it.kind, Adjust::Borrow(_)))
                        .unwrap_or(true)
                    {
                        // prefer reborrow to move
                        adjustments
                            .push(Adjustment { kind: Adjust::Deref(None), target: inner.store() });
                        adjustments.push(Adjustment::borrow(
                            self.interner(),
                            Mutability::Mut,
                            inner,
                            lt,
                        ))
                    }
                } else {
                    adjustments.push(Adjustment::borrow(
                        self.interner(),
                        Mutability::Mut,
                        derefed_callee,
                        self.table.next_region_var(),
                    ));
                }
            }
            FnTrait::Fn | FnTrait::AsyncFn => {
                if !matches!(derefed_callee.kind(), TyKind::Ref(_, _, Mutability::Not)) {
                    adjustments.push(Adjustment::borrow(
                        self.interner(),
                        Mutability::Not,
                        derefed_callee,
                        self.table.next_region_var(),
                    ));
                }
            }
        }
        let Some(trait_) = fn_x.get_id(self.lang_items) else {
            return;
        };
        let trait_data = trait_.trait_items(self.db);
        if let Some(func) = trait_data.method_by_name(&fn_x.method_name()) {
            let subst = GenericArgs::new_from_slice(&[
                callee_ty.into(),
                Ty::new_tup(self.interner(), params).into(),
            ]);
            self.write_method_resolution(tgt_expr, func, subst);
        }
    }

    fn infer_expr_array(&mut self, array: &Array, expected: &Expectation<'db>) -> Ty<'db> {
        let elem_ty = match expected
            .to_option(&mut self.table)
            .map(|t| self.table.try_structurally_resolve_type(t).kind())
        {
            Some(TyKind::Array(st, _) | TyKind::Slice(st)) => st,
            _ => self.table.next_ty_var(),
        };

        let krate = self.resolver.krate();

        let expected = Expectation::has_type(elem_ty);
        let (elem_ty, len) = match array {
            Array::ElementList { elements, .. } if elements.is_empty() => {
                (elem_ty, consteval::usize_const(self.db, Some(0), krate))
            }
            Array::ElementList { elements, .. } => {
                let mut coerce = CoerceMany::with_coercion_sites(elem_ty, elements);
                for &expr in elements.iter() {
                    let cur_elem_ty = self.infer_expr_inner(expr, &expected, ExprIsRead::Yes);
                    coerce.coerce(
                        self,
                        &ObligationCause::new(),
                        expr,
                        cur_elem_ty,
                        ExprIsRead::Yes,
                    );
                }
                (
                    coerce.complete(self),
                    consteval::usize_const(self.db, Some(elements.len() as u128), krate),
                )
            }
            &Array::Repeat { initializer, repeat } => {
                self.infer_expr_coerce(
                    initializer,
                    &Expectation::has_type(elem_ty),
                    ExprIsRead::Yes,
                );
                let usize = self.types.types.usize;
                let len = match self.body[repeat] {
                    Expr::Underscore => {
                        self.write_expr_ty(repeat, usize);
                        self.table.next_const_var()
                    }
                    _ => {
                        self.infer_expr(repeat, &Expectation::HasType(usize), ExprIsRead::Yes);
                        consteval::eval_to_const(repeat, self)
                    }
                };

                (elem_ty, len)
            }
        };
        // Try to evaluate unevaluated constant, and insert variable if is not possible.
        let len = self.table.insert_const_vars_shallow(len);
        Ty::new_array_with_const_len(self.interner(), elem_ty, len)
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
        coerce_many.coerce(self, &ObligationCause::new(), expr, return_expr_ty, ExprIsRead::Yes);
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
                        &ObligationCause::new(),
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
                self.unify(call_expr_ty, ret_ty);
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
            Ty::new_adt(
                self.interner(),
                box_id,
                GenericArgs::fill_with_defaults(
                    self.interner(),
                    box_id.into(),
                    [inner_ty.into()],
                    |_, id, _| self.table.next_var_for_param(id),
                ),
            )
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
        let coerce_ty = expected.coercion_target_type(&mut self.table);
        let g = self.resolver.update_to_inner_scope(self.db, self.owner, expr);

        let (break_ty, ty) =
            self.with_breakable_ctx(BreakableKind::Block, Some(coerce_ty), label, |this| {
                for stmt in statements {
                    match stmt {
                        Statement::Let { pat, type_ref, initializer, else_branch } => {
                            let decl_ty = type_ref
                                .as_ref()
                                .map(|&tr| this.make_body_ty(tr))
                                .unwrap_or_else(|| this.table.next_ty_var());

                            let ty = if let Some(expr) = initializer {
                                // If we have a subpattern that performs a read, we want to consider this
                                // to diverge for compatibility to support something like `let x: () = *never_ptr;`.
                                let target_is_read =
                                    if this.pat_guaranteed_to_constitute_read_for_never(*pat) {
                                        ExprIsRead::Yes
                                    } else {
                                        ExprIsRead::No
                                    };
                                let ty = if contains_explicit_ref_binding(this.body, *pat) {
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

                            let decl = DeclContext {
                                origin: DeclOrigin::LocalDecl { has_else: else_branch.is_some() },
                            };

                            this.infer_top_pat(*pat, ty, Some(decl));
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
                        this.table.new_maybe_never_var()
                    } else if let Some(t) = expected.only_has_type(&mut this.table) {
                        if this
                            .coerce(
                                expr.into(),
                                this.types.types.unit,
                                t,
                                AllowTwoPhase::No,
                                ExprIsRead::Yes,
                            )
                            .is_err()
                        {
                            this.result.type_mismatches.get_or_insert_default().insert(
                                expr.into(),
                                TypeMismatch {
                                    expected: t.store(),
                                    actual: this.types.types.unit.store(),
                                },
                            );
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
        receiver_ty: Ty<'db>,
        name: &Name,
    ) -> Option<(Ty<'db>, Either<FieldId, TupleFieldId>, Vec<Adjustment>, bool)> {
        let interner = self.interner();
        let mut autoderef = self.table.autoderef_with_tracking(receiver_ty);
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
                TyKind::Adt(adt, parameters) => match adt.def_id().0 {
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
            let is_visible = self.db.field_visibilities(field_id.parent)[field_id.local_id]
                .is_visible_from(self.db, self.resolver.module());
            if !is_visible {
                if private_field.is_none() {
                    private_field = Some((field_id, parameters));
                }
                return None;
            }
            let ty = self.db.field_types(field_id.parent)[field_id.local_id]
                .get()
                .instantiate(interner, parameters);
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
                    .instantiate(self.interner(), subst);
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
        let receiver_ty = self.table.structurally_resolve_type(receiver_ty);

        if name.is_missing() {
            // Bail out early, don't even try to look up field. Also, we don't issue an unresolved
            // field diagnostic because this is a syntax error rather than a semantic error.
            return self.err_ty();
        }

        match self.lookup_field(receiver_ty, name) {
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
        let args = self.table.fresh_args_for_item(def_id.into());
        let sig = self.db.callable_item_signature(def_id.into()).instantiate(self.interner(), args);
        let sig =
            self.infcx().instantiate_binder_with_fresh_vars(BoundRegionConversionTime::FnCall, sig);
        MethodCallee { def_id, args, sig }
    }

    fn infer_call(
        &mut self,
        tgt_expr: ExprId,
        callee: ExprId,
        args: &[ExprId],
        expected: &Expectation<'db>,
    ) -> Ty<'db> {
        let callee_ty = self.infer_expr(callee, &Expectation::none(), ExprIsRead::Yes);
        let callee_ty = self.table.try_structurally_resolve_type(callee_ty);
        let interner = self.interner();
        let mut derefs = InferenceContextAutoderef::new_from_inference_context(self, callee_ty);
        let (res, derefed_callee) = loop {
            let Some((callee_deref_ty, _)) = derefs.next() else {
                break (None, callee_ty);
            };
            if let Some(res) = derefs.ctx().table.callable_sig(callee_deref_ty, args.len()) {
                break (Some(res), callee_deref_ty);
            }
        };
        // if the function is unresolved, we use is_varargs=true to
        // suppress the arg count diagnostic here
        let is_varargs = derefed_callee.callable_sig(interner).is_some_and(|sig| sig.c_variadic())
            || res.is_none();
        let (param_tys, ret_ty) = match res {
            Some((func, params, ret_ty)) => {
                let infer_ok = derefs.adjust_steps_as_infer_ok();
                let mut adjustments = self.table.register_infer_ok(infer_ok);
                if let Some(fn_x) = func {
                    self.write_fn_trait_method_resolution(
                        fn_x,
                        derefed_callee,
                        &mut adjustments,
                        callee_ty,
                        &params,
                        tgt_expr,
                    );
                }
                if let TyKind::Closure(c, _) = self.table.resolve_completely(callee_ty).kind() {
                    self.add_current_closure_dependency(c.into());
                    self.deferred_closures.entry(c.into()).or_default().push((
                        derefed_callee,
                        callee_ty,
                        params.clone(),
                        tgt_expr,
                    ));
                }
                self.write_expr_adj(callee, adjustments.into_boxed_slice());
                (params, ret_ty)
            }
            None => {
                self.push_diagnostic(InferenceDiagnostic::ExpectedFunction {
                    call_expr: tgt_expr,
                    found: callee_ty.store(),
                });
                (Vec::new(), Ty::new_error(interner, ErrorGuaranteed))
            }
        };
        let indices_to_skip = self.check_legacy_const_generics(derefed_callee, args);
        self.check_call(
            tgt_expr,
            args,
            callee_ty,
            &param_tys,
            ret_ty,
            &indices_to_skip,
            is_varargs,
            expected,
        )
    }

    fn check_call(
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
        self.register_obligations_for_call(callee_ty);

        self.check_call_arguments(
            tgt_expr,
            param_tys,
            ret_ty,
            expected,
            args,
            indices_to_skip,
            is_varargs,
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
        let receiver_ty = self.table.try_structurally_resolve_type(receiver_ty);

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
                let field_with_same_name_exists = match self.lookup_field(receiver_ty, method_name)
                {
                    Some((ty, field_id, adjustments, _public)) => {
                        self.write_expr_adj(receiver, adjustments.into_boxed_slice());
                        self.result.field_resolutions.insert(tgt_expr, field_id);
                        Some(ty)
                    }
                    None => None,
                };

                let assoc_func_with_same_name = self.with_method_resolution(|ctx| {
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
                    Some((callee_ty, sig, strip_first)) => self.check_call(
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

        self.check_call_arguments(tgt_expr, param_tys, ret_ty, expected, args, &[], sig.c_variadic);
        ret_ty
    }

    /// Generic function that factors out common logic from function calls,
    /// method calls and overloaded operators.
    pub(in super::super) fn check_call_arguments(
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
    ) {
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
                        let origin = ObligationCause::new();
                        ocx.sup(&origin, self.table.param_env, expected_output, formal_output)?;
                        if !ocx.try_evaluate_obligations().is_empty() {
                            return Err(TypeError::Mismatch);
                        }

                        // Record all the argument types, with the args
                        // produced from the above subtyping unification.
                        Ok(Some(
                            formal_input_tys
                                .iter()
                                .map(|&ty| self.table.infer_ctxt.resolve_vars_if_possible(ty))
                                .collect(),
                        ))
                    })
                    .ok()
            })
            .unwrap_or_default();

        // If there are no external expectations at the call site, just use the types from the function defn
        let expected_input_tys = if let Some(expected_input_tys) = &expected_input_tys {
            assert_eq!(expected_input_tys.len(), formal_input_tys.len());
            expected_input_tys
        } else {
            formal_input_tys
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
                .coerce(
                    provided_arg.into(),
                    checked_ty,
                    coerced_ty,
                    AllowTwoPhase::Yes,
                    ExprIsRead::Yes,
                )
                .err();
            if coerce_error.is_some() {
                return Err((coerce_error, coerced_ty, checked_ty));
            }

            // 3. Check if the formal type is actually equal to the checked one
            //    and register any such obligations for future type checks.
            let formal_ty_error = this
                .table
                .infer_ctxt
                .at(&ObligationCause::new(), this.table.param_env)
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
                let is_closure = if let Expr::Closure { closure_kind, .. } = self.body[*arg] {
                    !matches!(closure_kind, ClosureKind::Coroutine(_))
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
                    self.result.type_mismatches.get_or_insert_default().insert(
                        (*arg).into(),
                        TypeMismatch { expected: expected.store(), actual: found.store() },
                    );
                }
            }
        }

        if !args_count_matches {}
    }

    fn register_obligations_for_call(&mut self, callable_ty: Ty<'db>) {
        let callable_ty = self.table.try_structurally_resolve_type(callable_ty);
        if let TyKind::FnDef(fn_def, parameters) = callable_ty.kind() {
            let generic_predicates = GenericPredicates::query_all(
                self.db,
                GenericDefId::from_callable(self.db, fn_def.0),
            );
            let param_env = self.table.param_env;
            self.table.register_predicates(clauses_as_obligations(
                generic_predicates.iter_instantiated_copied(self.interner(), parameters.as_slice()),
                ObligationCause::new(),
                param_env,
            ));
            // add obligation for trait implementation, if this is a trait method
            match fn_def.0 {
                CallableDefId::FunctionId(f) => {
                    if let ItemContainerId::TraitId(trait_) = f.lookup(self.db).container {
                        // construct a TraitRef
                        let trait_params_len = generics(self.db, trait_.into()).len();
                        let substs =
                            GenericArgs::new_from_slice(&parameters.as_slice()[..trait_params_len]);
                        self.table.register_predicate(Obligation::new(
                            self.interner(),
                            ObligationCause::new(),
                            self.table.param_env,
                            TraitRef::new_from_args(self.interner(), trait_.into(), substs),
                        ));
                    }
                }
                CallableDefId::StructId(_) | CallableDefId::EnumVariantId(_) => {}
            }
        }
    }

    /// Returns the argument indices to skip.
    fn check_legacy_const_generics(&mut self, callee: Ty<'db>, args: &[ExprId]) -> Box<[u32]> {
        let (func, _subst) = match callee.kind() {
            TyKind::FnDef(callable, subst) => {
                let func = match callable.0 {
                    CallableDefId::FunctionId(f) => f,
                    _ => return Default::default(),
                };
                (func, subst)
            }
            _ => return Default::default(),
        };

        let data = self.db.function_signature(func);
        let Some(legacy_const_generics_indices) = data.legacy_const_generics_indices(self.db, func)
        else {
            return Default::default();
        };
        let mut legacy_const_generics_indices = Box::<[u32]>::from(legacy_const_generics_indices);

        // only use legacy const generics if the param count matches with them
        if data.params.len() + legacy_const_generics_indices.len() != args.len() {
            if args.len() <= data.params.len() {
                return Default::default();
            } else {
                // there are more parameters than there should be without legacy
                // const params; use them
                legacy_const_generics_indices.sort_unstable();
                return legacy_const_generics_indices;
            }
        }

        // check legacy const parameters
        for arg_idx in legacy_const_generics_indices.iter().copied() {
            if arg_idx >= args.len() as u32 {
                continue;
            }
            let expected = Expectation::none(); // FIXME use actual const ty, when that is lowered correctly
            self.infer_expr(args[arg_idx as usize], &expected, ExprIsRead::Yes);
            // FIXME: evaluate and unify with the const
        }
        legacy_const_generics_indices.sort_unstable();
        legacy_const_generics_indices
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
