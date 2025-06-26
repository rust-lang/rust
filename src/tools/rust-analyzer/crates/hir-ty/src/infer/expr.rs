//! Type inference for expressions.

use std::{
    iter::{repeat, repeat_with},
    mem,
};

use chalk_ir::{DebruijnIndex, Mutability, TyVariableKind, cast::Cast};
use either::Either;
use hir_def::{
    BlockId, FieldId, GenericDefId, GenericParamId, ItemContainerId, Lookup, TupleFieldId, TupleId,
    expr_store::path::{GenericArg, GenericArgs, Path},
    hir::{
        ArithOp, Array, AsmOperand, AsmOptions, BinaryOp, Expr, ExprId, ExprOrPatId, LabelId,
        Literal, Pat, PatId, Statement, UnaryOp, generics::GenericParamDataRef,
    },
    lang_item::{LangItem, LangItemTarget},
    resolver::ValueNs,
};
use hir_expand::name::Name;
use intern::sym;
use stdx::always;
use syntax::ast::RangeOp;

use crate::{
    Adjust, Adjustment, AdtId, AutoBorrow, Binders, CallableDefId, CallableSig, DeclContext,
    DeclOrigin, IncorrectGenericsLenKind, Interner, Rawness, Scalar, Substitution,
    TraitEnvironment, TraitRef, Ty, TyBuilder, TyExt, TyKind,
    autoderef::{Autoderef, builtin_deref, deref_by_trait},
    consteval,
    generics::generics,
    infer::{
        BreakableKind,
        coerce::{CoerceMany, CoerceNever, CoercionCause},
        find_continuable,
        pat::contains_explicit_ref_binding,
    },
    lang_items::lang_items_for_bin_op,
    lower::{
        LifetimeElisionKind, ParamLoweringMode, lower_to_chalk_mutability,
        path::{GenericArgsLowerer, TypeLikeConst, substs_from_args_and_bindings},
    },
    mapping::{ToChalk, from_chalk},
    method_resolution::{self, VisibleFromModule},
    primitive::{self, UintTy},
    static_lifetime, to_chalk_trait_id,
    traits::FnTrait,
};

use super::{
    BreakableContext, Diverges, Expectation, InferenceContext, InferenceDiagnostic, TypeMismatch,
    cast::CastCheck, coerce::auto_deref_adjust_steps, find_breakable,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExprIsRead {
    Yes,
    No,
}

impl InferenceContext<'_> {
    pub(crate) fn infer_expr(
        &mut self,
        tgt_expr: ExprId,
        expected: &Expectation,
        is_read: ExprIsRead,
    ) -> Ty {
        let ty = self.infer_expr_inner(tgt_expr, expected, is_read);
        if let Some(expected_ty) = expected.only_has_type(&mut self.table) {
            let could_unify = self.unify(&ty, &expected_ty);
            if !could_unify {
                self.result.type_mismatches.insert(
                    tgt_expr.into(),
                    TypeMismatch { expected: expected_ty, actual: ty.clone() },
                );
            }
        }
        ty
    }

    pub(crate) fn infer_expr_no_expect(&mut self, tgt_expr: ExprId, is_read: ExprIsRead) -> Ty {
        self.infer_expr_inner(tgt_expr, &Expectation::None, is_read)
    }

    /// Infer type of expression with possibly implicit coerce to the expected type.
    /// Return the type after possible coercion.
    pub(super) fn infer_expr_coerce(
        &mut self,
        expr: ExprId,
        expected: &Expectation,
        is_read: ExprIsRead,
    ) -> Ty {
        let ty = self.infer_expr_inner(expr, expected, is_read);
        if let Some(target) = expected.only_has_type(&mut self.table) {
            let coerce_never = if self.expr_guaranteed_to_constitute_read_for_never(expr, is_read) {
                CoerceNever::Yes
            } else {
                CoerceNever::No
            };
            match self.coerce(Some(expr), &ty, &target, coerce_never) {
                Ok(res) => res,
                Err(_) => {
                    self.result.type_mismatches.insert(
                        expr.into(),
                        TypeMismatch { expected: target.clone(), actual: ty.clone() },
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

    fn infer_expr_coerce_never(
        &mut self,
        expr: ExprId,
        expected: &Expectation,
        is_read: ExprIsRead,
    ) -> Ty {
        let ty = self.infer_expr_inner(expr, expected, is_read);
        // While we don't allow *arbitrary* coercions here, we *do* allow
        // coercions from `!` to `expected`.
        if ty.is_never() {
            if let Some(adjustments) = self.result.expr_adjustments.get(&expr) {
                return if let [Adjustment { kind: Adjust::NeverToAny, target }] = &**adjustments {
                    target.clone()
                } else {
                    self.err_ty()
                };
            }

            if let Some(target) = expected.only_has_type(&mut self.table) {
                self.coerce(Some(expr), &ty, &target, CoerceNever::Yes)
                    .expect("never-to-any coercion should always succeed")
            } else {
                ty
            }
        } else {
            if let Some(expected_ty) = expected.only_has_type(&mut self.table) {
                let could_unify = self.unify(&ty, &expected_ty);
                if !could_unify {
                    self.result.type_mismatches.insert(
                        expr.into(),
                        TypeMismatch { expected: expected_ty, actual: ty.clone() },
                    );
                }
            }
            ty
        }
    }

    fn infer_expr_inner(
        &mut self,
        tgt_expr: ExprId,
        expected: &Expectation,
        is_read: ExprIsRead,
    ) -> Ty {
        self.db.unwind_if_revision_cancelled();

        let ty = match &self.body[tgt_expr] {
            Expr::Missing => self.err_ty(),
            &Expr::If { condition, then_branch, else_branch } => {
                let expected = &expected.adjust_for_branches(&mut self.table);
                self.infer_expr_coerce_never(
                    condition,
                    &Expectation::HasType(self.result.standard_types.bool_.clone()),
                    ExprIsRead::Yes,
                );

                let condition_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);

                let then_ty = self.infer_expr_inner(then_branch, expected, ExprIsRead::Yes);
                let then_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
                let mut coerce = CoerceMany::new(expected.coercion_target_type(&mut self.table));
                coerce.coerce(self, Some(then_branch), &then_ty, CoercionCause::Expr(then_branch));
                match else_branch {
                    Some(else_branch) => {
                        let else_ty = self.infer_expr_inner(else_branch, expected, ExprIsRead::Yes);
                        let else_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
                        coerce.coerce(
                            self,
                            Some(else_branch),
                            &else_ty,
                            CoercionCause::Expr(else_branch),
                        );
                        self.diverges = condition_diverges | then_diverges & else_diverges;
                    }
                    None => {
                        coerce.coerce_forced_unit(self, CoercionCause::Expr(tgt_expr));
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
                    &input_ty,
                    Some(DeclContext { origin: DeclOrigin::LetExpr }),
                );
                self.result.standard_types.bool_.clone()
            }
            Expr::Block { statements, tail, label, id } => {
                self.infer_block(tgt_expr, *id, statements, *tail, *label, expected)
            }
            Expr::Unsafe { id, statements, tail } => {
                self.infer_block(tgt_expr, *id, statements, *tail, None, expected)
            }
            Expr::Const(id) => {
                self.with_breakable_ctx(BreakableKind::Border, None, None, |this| {
                    this.infer_expr(*id, expected, ExprIsRead::Yes)
                })
                .1
            }
            Expr::Async { id, statements, tail } => {
                self.infer_async_block(tgt_expr, id, statements, tail)
            }
            &Expr::Loop { body, label } => {
                // FIXME: should be:
                // let ty = expected.coercion_target_type(&mut self.table);
                let ty = self.table.new_type_var();
                let (breaks, ()) =
                    self.with_breakable_ctx(BreakableKind::Loop, Some(ty), label, |this| {
                        this.infer_expr(
                            body,
                            &Expectation::HasType(TyBuilder::unit()),
                            ExprIsRead::Yes,
                        );
                    });

                match breaks {
                    Some(breaks) => {
                        self.diverges = Diverges::Maybe;
                        breaks
                    }
                    None => self.result.standard_types.never.clone(),
                }
            }
            Expr::Closure { body, args, ret_type, arg_types, closure_kind, capture_by: _ } => self
                .infer_closure(body, args, ret_type, arg_types, *closure_kind, tgt_expr, expected),
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
                let scrutinee_is_read = arms
                    .iter()
                    .all(|arm| self.pat_guaranteed_to_constitute_read_for_never(arm.pat));
                let scrutinee_is_read =
                    if scrutinee_is_read { ExprIsRead::Yes } else { ExprIsRead::No };
                let input_ty = self.infer_expr(*expr, &Expectation::none(), scrutinee_is_read);

                if arms.is_empty() {
                    self.diverges = Diverges::Always;
                    self.result.standard_types.never.clone()
                } else {
                    let matchee_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
                    let mut all_arms_diverge = Diverges::Always;
                    for arm in arms.iter() {
                        let input_ty = self.resolve_ty_shallow(&input_ty);
                        self.infer_top_pat(arm.pat, &input_ty, None);
                    }

                    let expected = expected.adjust_for_branches(&mut self.table);
                    let result_ty = match &expected {
                        // We don't coerce to `()` so that if the match expression is a
                        // statement it's branches can have any consistent type.
                        Expectation::HasType(ty) if *ty != self.result.standard_types.unit => {
                            ty.clone()
                        }
                        _ => self.table.new_type_var(),
                    };
                    let mut coerce = CoerceMany::new(result_ty);

                    for arm in arms.iter() {
                        if let Some(guard_expr) = arm.guard {
                            self.diverges = Diverges::Maybe;
                            self.infer_expr_coerce_never(
                                guard_expr,
                                &Expectation::HasType(self.result.standard_types.bool_.clone()),
                                ExprIsRead::Yes,
                            );
                        }
                        self.diverges = Diverges::Maybe;

                        let arm_ty = self.infer_expr_inner(arm.expr, &expected, ExprIsRead::Yes);
                        all_arms_diverge &= self.diverges;
                        coerce.coerce(self, Some(arm.expr), &arm_ty, CoercionCause::Expr(arm.expr));
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
                self.result.standard_types.never.clone()
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
                    TyBuilder::unit()
                };

                match find_breakable(&mut self.breakables, label) {
                    Some(ctxt) => match ctxt.coerce.take() {
                        Some(mut coerce) => {
                            let cause = match expr {
                                Some(expr) => CoercionCause::Expr(expr),
                                None => CoercionCause::Expr(tgt_expr),
                            };
                            coerce.coerce(self, expr, &val_ty, cause);

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
                self.result.standard_types.never.clone()
            }
            &Expr::Return { expr } => self.infer_expr_return(tgt_expr, expr),
            &Expr::Become { expr } => self.infer_expr_become(expr),
            Expr::Yield { expr } => {
                if let Some((resume_ty, yield_ty)) = self.resume_yield_tys.clone() {
                    if let Some(expr) = expr {
                        self.infer_expr_coerce(
                            *expr,
                            &Expectation::has_type(yield_ty),
                            ExprIsRead::Yes,
                        );
                    } else {
                        let unit = self.result.standard_types.unit.clone();
                        let _ = self.coerce(Some(tgt_expr), &unit, &yield_ty, CoerceNever::Yes);
                    }
                    resume_ty
                } else {
                    // FIXME: report error (yield expr in non-coroutine)
                    self.result.standard_types.unknown.clone()
                }
            }
            Expr::Yeet { expr } => {
                if let &Some(expr) = expr {
                    self.infer_expr_no_expect(expr, ExprIsRead::Yes);
                }
                self.result.standard_types.never.clone()
            }
            Expr::RecordLit { path, fields, spread, .. } => {
                let (ty, def_id) = self.resolve_variant(tgt_expr.into(), path.as_deref(), false);

                if let Some(t) = expected.only_has_type(&mut self.table) {
                    self.unify(&ty, &t);
                }

                let substs = ty
                    .as_adt()
                    .map(|(_, s)| s.clone())
                    .unwrap_or_else(|| Substitution::empty(Interner));
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
                                field_types[it].clone().substitute(Interner, &substs)
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
                if let Some(expr) = spread {
                    self.infer_expr(*expr, &Expectation::has_type(ty.clone()), ExprIsRead::Yes);
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
                let expr_ty = self.infer_expr(
                    *expr,
                    &Expectation::Castable(cast_ty.clone()),
                    ExprIsRead::Yes,
                );
                self.deferred_cast_checks.push(CastCheck::new(
                    tgt_expr,
                    *expr,
                    expr_ty,
                    cast_ty.clone(),
                ));
                cast_ty
            }
            Expr::Ref { expr, rawness, mutability } => {
                let mutability = lower_to_chalk_mutability(*mutability);
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
                    Expectation::rvalue_hint(self, Ty::clone(exp_inner))
                } else {
                    Expectation::none()
                };
                let inner_ty = self.infer_expr_inner(*expr, &expectation, ExprIsRead::Yes);
                match rawness {
                    Rawness::RawPtr => TyKind::Raw(mutability, inner_ty),
                    Rawness::Ref => {
                        let lt = self.table.new_lifetime_var();
                        TyKind::Ref(mutability, lt, inner_ty)
                    }
                }
                .intern(Interner)
            }
            &Expr::Box { expr } => self.infer_expr_box(expr, expected),
            Expr::UnaryOp { expr, op } => {
                let inner_ty = self.infer_expr_inner(*expr, &Expectation::none(), ExprIsRead::Yes);
                let inner_ty = self.resolve_ty_shallow(&inner_ty);
                // FIXME: Note down method resolution her
                match op {
                    UnaryOp::Deref => {
                        if let Some(deref_trait) = self.resolve_lang_trait(LangItem::Deref) {
                            if let Some(deref_fn) = deref_trait
                                .trait_items(self.db)
                                .method_by_name(&Name::new_symbol_root(sym::deref))
                            {
                                // FIXME: this is wrong in multiple ways, subst is empty, and we emit it even for builtin deref (note that
                                // the mutability is not wrong, and will be fixed in `self.infer_mut`).
                                self.write_method_resolution(
                                    tgt_expr,
                                    deref_fn,
                                    Substitution::empty(Interner),
                                );
                            }
                        }
                        if let Some(derefed) = builtin_deref(self.table.db, &inner_ty, true) {
                            self.resolve_ty_shallow(derefed)
                        } else {
                            deref_by_trait(&mut self.table, inner_ty, false)
                                .unwrap_or_else(|| self.err_ty())
                        }
                    }
                    UnaryOp::Neg => {
                        match inner_ty.kind(Interner) {
                            // Fast path for builtins
                            TyKind::Scalar(Scalar::Int(_) | Scalar::Uint(_) | Scalar::Float(_))
                            | TyKind::InferenceVar(
                                _,
                                TyVariableKind::Integer | TyVariableKind::Float,
                            ) => inner_ty,
                            // Otherwise we resolve via the std::ops::Neg trait
                            _ => self
                                .resolve_associated_type(inner_ty, self.resolve_ops_neg_output()),
                        }
                    }
                    UnaryOp::Not => {
                        match inner_ty.kind(Interner) {
                            // Fast path for builtins
                            TyKind::Scalar(Scalar::Bool | Scalar::Int(_) | Scalar::Uint(_))
                            | TyKind::InferenceVar(_, TyVariableKind::Integer) => inner_ty,
                            // Otherwise we resolve via the std::ops::Not trait
                            _ => self
                                .resolve_associated_type(inner_ty, self.resolve_ops_not_output()),
                        }
                    }
                }
            }
            Expr::BinaryOp { lhs, rhs, op } => match op {
                Some(BinaryOp::LogicOp(_)) => {
                    let bool_ty = self.result.standard_types.bool_.clone();
                    self.infer_expr_coerce(
                        *lhs,
                        &Expectation::HasType(bool_ty.clone()),
                        ExprIsRead::Yes,
                    );
                    let lhs_diverges = self.diverges;
                    self.infer_expr_coerce(
                        *rhs,
                        &Expectation::HasType(bool_ty.clone()),
                        ExprIsRead::Yes,
                    );
                    // Depending on the LHS' value, the RHS can never execute.
                    self.diverges = lhs_diverges;
                    bool_ty
                }
                Some(op) => self.infer_overloadable_binop(*lhs, *op, *rhs, tgt_expr),
                _ => self.err_ty(),
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
                    Pat::Path(path) => Some(self.infer_expr_path(path, target.into(), tgt_expr)),
                    _ => None,
                };

                if let Some(lhs_ty) = lhs_ty {
                    self.write_pat_ty(target, lhs_ty.clone());
                    self.infer_expr_coerce(value, &Expectation::has_type(lhs_ty), ExprIsRead::No);
                } else {
                    let rhs_ty = self.infer_expr(value, &Expectation::none(), ExprIsRead::Yes);
                    let resolver_guard =
                        self.resolver.update_to_inner_scope(self.db, self.owner, tgt_expr);
                    self.inside_assignment = true;
                    self.infer_top_pat(target, &rhs_ty, None);
                    self.inside_assignment = false;
                    self.resolver.reset_to_guard(resolver_guard);
                }
                self.result.standard_types.unit.clone()
            }
            Expr::Range { lhs, rhs, range_type } => {
                let lhs_ty =
                    lhs.map(|e| self.infer_expr_inner(e, &Expectation::none(), ExprIsRead::Yes));
                let rhs_expect = lhs_ty
                    .as_ref()
                    .map_or_else(Expectation::none, |ty| Expectation::has_type(ty.clone()));
                let rhs_ty = rhs.map(|e| self.infer_expr(e, &rhs_expect, ExprIsRead::Yes));
                match (range_type, lhs_ty, rhs_ty) {
                    (RangeOp::Exclusive, None, None) => match self.resolve_range_full() {
                        Some(adt) => TyBuilder::adt(self.db, adt).build(),
                        None => self.err_ty(),
                    },
                    (RangeOp::Exclusive, None, Some(ty)) => match self.resolve_range_to() {
                        Some(adt) => TyBuilder::adt(self.db, adt).push(ty).build(),
                        None => self.err_ty(),
                    },
                    (RangeOp::Inclusive, None, Some(ty)) => {
                        match self.resolve_range_to_inclusive() {
                            Some(adt) => TyBuilder::adt(self.db, adt).push(ty).build(),
                            None => self.err_ty(),
                        }
                    }
                    (RangeOp::Exclusive, Some(_), Some(ty)) => match self.resolve_range() {
                        Some(adt) => TyBuilder::adt(self.db, adt).push(ty).build(),
                        None => self.err_ty(),
                    },
                    (RangeOp::Inclusive, Some(_), Some(ty)) => {
                        match self.resolve_range_inclusive() {
                            Some(adt) => TyBuilder::adt(self.db, adt).push(ty).build(),
                            None => self.err_ty(),
                        }
                    }
                    (RangeOp::Exclusive, Some(ty), None) => match self.resolve_range_from() {
                        Some(adt) => TyBuilder::adt(self.db, adt).push(ty).build(),
                        None => self.err_ty(),
                    },
                    (RangeOp::Inclusive, _, None) => self.err_ty(),
                }
            }
            Expr::Index { base, index } => {
                let base_ty = self.infer_expr_inner(*base, &Expectation::none(), ExprIsRead::Yes);
                let index_ty = self.infer_expr(*index, &Expectation::none(), ExprIsRead::Yes);

                if let Some(index_trait) = self.resolve_lang_trait(LangItem::Index) {
                    let canonicalized = self.canonicalize(base_ty.clone());
                    let receiver_adjustments = method_resolution::resolve_indexing_op(
                        self.db,
                        self.table.trait_env.clone(),
                        canonicalized,
                        index_trait,
                    );
                    let (self_ty, mut adj) = receiver_adjustments
                        .map_or((self.err_ty(), Vec::new()), |adj| {
                            adj.apply(&mut self.table, base_ty)
                        });

                    // mutability will be fixed up in `InferenceContext::infer_mut`;
                    adj.push(Adjustment::borrow(
                        Mutability::Not,
                        self_ty.clone(),
                        self.table.new_lifetime_var(),
                    ));
                    self.write_expr_adj(*base, adj.into_boxed_slice());
                    if let Some(func) = index_trait
                        .trait_items(self.db)
                        .method_by_name(&Name::new_symbol_root(sym::index))
                    {
                        let subst = TyBuilder::subst_for_def(self.db, index_trait, None);
                        if subst.remaining() != 2 {
                            return self.err_ty();
                        }
                        let subst = subst.push(self_ty.clone()).push(index_ty.clone()).build();
                        self.write_method_resolution(tgt_expr, func, subst);
                    }
                    let assoc = self.resolve_ops_index_output();
                    self.resolve_associated_type_with_params(
                        self_ty,
                        assoc,
                        &[index_ty.cast(Interner)],
                    )
                } else {
                    self.err_ty()
                }
            }
            Expr::Tuple { exprs, .. } => {
                let mut tys = match expected
                    .only_has_type(&mut self.table)
                    .as_ref()
                    .map(|t| t.kind(Interner))
                {
                    Some(TyKind::Tuple(_, substs)) => substs
                        .iter(Interner)
                        .map(|a| a.assert_ty_ref(Interner).clone())
                        .chain(repeat_with(|| self.table.new_type_var()))
                        .take(exprs.len())
                        .collect::<Vec<_>>(),
                    _ => (0..exprs.len()).map(|_| self.table.new_type_var()).collect(),
                };

                for (expr, ty) in exprs.iter().zip(tys.iter_mut()) {
                    *ty = self.infer_expr_coerce(
                        *expr,
                        &Expectation::has_type(ty.clone()),
                        ExprIsRead::Yes,
                    );
                }

                TyKind::Tuple(tys.len(), Substitution::from_iter(Interner, tys)).intern(Interner)
            }
            Expr::Array(array) => self.infer_expr_array(array, expected),
            Expr::Literal(lit) => match lit {
                Literal::Bool(..) => self.result.standard_types.bool_.clone(),
                Literal::String(..) => {
                    TyKind::Ref(Mutability::Not, static_lifetime(), TyKind::Str.intern(Interner))
                        .intern(Interner)
                }
                Literal::ByteString(bs) => {
                    let byte_type = TyKind::Scalar(Scalar::Uint(UintTy::U8)).intern(Interner);

                    let len = consteval::usize_const(
                        self.db,
                        Some(bs.len() as u128),
                        self.resolver.krate(),
                    );

                    let array_type = TyKind::Array(byte_type, len).intern(Interner);
                    TyKind::Ref(Mutability::Not, static_lifetime(), array_type).intern(Interner)
                }
                Literal::CString(..) => TyKind::Ref(
                    Mutability::Not,
                    static_lifetime(),
                    self.resolve_lang_item(LangItem::CStr)
                        .and_then(LangItemTarget::as_struct)
                        .map_or_else(
                            || self.err_ty(),
                            |strukt| {
                                TyKind::Adt(AdtId(strukt.into()), Substitution::empty(Interner))
                                    .intern(Interner)
                            },
                        ),
                )
                .intern(Interner),
                Literal::Char(..) => TyKind::Scalar(Scalar::Char).intern(Interner),
                Literal::Int(_v, ty) => match ty {
                    Some(int_ty) => {
                        TyKind::Scalar(Scalar::Int(primitive::int_ty_from_builtin(*int_ty)))
                            .intern(Interner)
                    }
                    None => {
                        let expected_ty = expected.to_option(&mut self.table);
                        let opt_ty = match expected_ty.as_ref().map(|it| it.kind(Interner)) {
                            Some(TyKind::Scalar(Scalar::Int(_) | Scalar::Uint(_))) => expected_ty,
                            Some(TyKind::Scalar(Scalar::Char)) => {
                                Some(TyKind::Scalar(Scalar::Uint(UintTy::U8)).intern(Interner))
                            }
                            Some(TyKind::Raw(..) | TyKind::FnDef(..) | TyKind::Function(..)) => {
                                Some(TyKind::Scalar(Scalar::Uint(UintTy::Usize)).intern(Interner))
                            }
                            _ => None,
                        };
                        opt_ty.unwrap_or_else(|| self.table.new_integer_var())
                    }
                },
                Literal::Uint(_v, ty) => match ty {
                    Some(int_ty) => {
                        TyKind::Scalar(Scalar::Uint(primitive::uint_ty_from_builtin(*int_ty)))
                            .intern(Interner)
                    }
                    None => {
                        let expected_ty = expected.to_option(&mut self.table);
                        let opt_ty = match expected_ty.as_ref().map(|it| it.kind(Interner)) {
                            Some(TyKind::Scalar(Scalar::Int(_) | Scalar::Uint(_))) => expected_ty,
                            Some(TyKind::Scalar(Scalar::Char)) => {
                                Some(TyKind::Scalar(Scalar::Uint(UintTy::U8)).intern(Interner))
                            }
                            Some(TyKind::Raw(..) | TyKind::FnDef(..) | TyKind::Function(..)) => {
                                Some(TyKind::Scalar(Scalar::Uint(UintTy::Usize)).intern(Interner))
                            }
                            _ => None,
                        };
                        opt_ty.unwrap_or_else(|| self.table.new_integer_var())
                    }
                },
                Literal::Float(_v, ty) => match ty {
                    Some(float_ty) => {
                        TyKind::Scalar(Scalar::Float(primitive::float_ty_from_builtin(*float_ty)))
                            .intern(Interner)
                    }
                    None => {
                        let opt_ty = expected.to_option(&mut self.table).filter(|ty| {
                            matches!(ty.kind(Interner), TyKind::Scalar(Scalar::Float(_)))
                        });
                        opt_ty.unwrap_or_else(|| self.table.new_float_var())
                    }
                },
            },
            Expr::Underscore => {
                // Underscore expression is an error, we render a specialized diagnostic
                // to let the user know what type is expected though.
                let expected = expected.to_option(&mut self.table).unwrap_or_else(|| self.err_ty());
                self.push_diagnostic(InferenceDiagnostic::TypedHole {
                    expr: tgt_expr,
                    expected: expected.clone(),
                });
                expected
            }
            Expr::OffsetOf(_) => TyKind::Scalar(Scalar::Uint(UintTy::Usize)).intern(Interner),
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
                        let ty = this.resolve_ty_shallow(&ty);
                        match ty.kind(Interner) {
                            TyKind::FnDef(def, parameters) => {
                                let fnptr_ty = TyKind::Function(
                                    CallableSig::from_def(this.db, *def, parameters).to_fn_ptr(),
                                )
                                .intern(Interner);
                                _ = this.coerce(Some(expr), &ty, &fnptr_ty, CoerceNever::Yes);
                            }
                            TyKind::Ref(mutbl, _, base_ty) => {
                                let ptr_ty = TyKind::Raw(*mutbl, base_ty.clone()).intern(Interner);
                                _ = this.coerce(Some(expr), &ty, &ptr_ty, CoerceNever::Yes);
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
                            &Expectation::HasType(self.result.standard_types.unit.clone()),
                            ExprIsRead::No,
                        );
                    }
                    AsmOperand::Const(expr) => {
                        self.infer_expr(expr, &Expectation::None, ExprIsRead::No);
                    }
                    // FIXME: `sym` should report for things that are not functions or statics.
                    AsmOperand::Sym(_) => (),
                });
                if diverge {
                    self.result.standard_types.never.clone()
                } else {
                    self.result.standard_types.unit.clone()
                }
            }
        };
        // use a new type variable if we got unknown here
        let ty = self.insert_type_vars_shallow(ty);
        self.write_expr_ty(tgt_expr, ty.clone());
        if self.resolve_ty_shallow(&ty).is_never()
            && self.expr_guaranteed_to_constitute_read_for_never(tgt_expr, is_read)
        {
            // Any expression that produces a value of type `!` must have diverged
            self.diverges = Diverges::Always;
        }
        ty
    }

    fn infer_expr_path(&mut self, path: &Path, id: ExprOrPatId, scope_id: ExprId) -> Ty {
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

    fn infer_async_block(
        &mut self,
        tgt_expr: ExprId,
        id: &Option<BlockId>,
        statements: &[Statement],
        tail: &Option<ExprId>,
    ) -> Ty {
        let ret_ty = self.table.new_type_var();
        let prev_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
        let prev_ret_ty = mem::replace(&mut self.return_ty, ret_ty.clone());
        let prev_ret_coercion = self.return_coercion.replace(CoerceMany::new(ret_ty.clone()));

        // FIXME: We should handle async blocks like we handle closures
        let expected = &Expectation::has_type(ret_ty);
        let (_, inner_ty) = self.with_breakable_ctx(BreakableKind::Border, None, None, |this| {
            let ty = this.infer_block(tgt_expr, *id, statements, *tail, None, expected);
            if let Some(target) = expected.only_has_type(&mut this.table) {
                match this.coerce(Some(tgt_expr), &ty, &target, CoerceNever::Yes) {
                    Ok(res) => res,
                    Err(_) => {
                        this.result.type_mismatches.insert(
                            tgt_expr.into(),
                            TypeMismatch { expected: target.clone(), actual: ty.clone() },
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
        inner_ty: Ty,
        tgt_expr: ExprId,
    ) -> Ty {
        // Use the first type parameter as the output type of future.
        // existential type AsyncBlockImplTrait<InnerType>: Future<Output = InnerType>
        let impl_trait_id = crate::ImplTraitId::AsyncBlockTypeImplTrait(self.owner, tgt_expr);
        let opaque_ty_id = self.db.intern_impl_trait_id(impl_trait_id).into();
        TyKind::OpaqueType(opaque_ty_id, Substitution::from1(Interner, inner_ty)).intern(Interner)
    }

    pub(crate) fn write_fn_trait_method_resolution(
        &mut self,
        fn_x: FnTrait,
        derefed_callee: &Ty,
        adjustments: &mut Vec<Adjustment>,
        callee_ty: &Ty,
        params: &[Ty],
        tgt_expr: ExprId,
    ) {
        match fn_x {
            FnTrait::FnOnce | FnTrait::AsyncFnOnce => (),
            FnTrait::FnMut | FnTrait::AsyncFnMut => {
                if let TyKind::Ref(Mutability::Mut, lt, inner) = derefed_callee.kind(Interner) {
                    if adjustments
                        .last()
                        .map(|it| matches!(it.kind, Adjust::Borrow(_)))
                        .unwrap_or(true)
                    {
                        // prefer reborrow to move
                        adjustments
                            .push(Adjustment { kind: Adjust::Deref(None), target: inner.clone() });
                        adjustments.push(Adjustment::borrow(
                            Mutability::Mut,
                            inner.clone(),
                            lt.clone(),
                        ))
                    }
                } else {
                    adjustments.push(Adjustment::borrow(
                        Mutability::Mut,
                        derefed_callee.clone(),
                        self.table.new_lifetime_var(),
                    ));
                }
            }
            FnTrait::Fn | FnTrait::AsyncFn => {
                if !matches!(derefed_callee.kind(Interner), TyKind::Ref(Mutability::Not, _, _)) {
                    adjustments.push(Adjustment::borrow(
                        Mutability::Not,
                        derefed_callee.clone(),
                        self.table.new_lifetime_var(),
                    ));
                }
            }
        }
        let Some(trait_) = fn_x.get_id(self.db, self.table.trait_env.krate) else {
            return;
        };
        let trait_data = trait_.trait_items(self.db);
        if let Some(func) = trait_data.method_by_name(&fn_x.method_name()) {
            let subst = TyBuilder::subst_for_def(self.db, trait_, None)
                .push(callee_ty.clone())
                .push(TyBuilder::tuple_with(params.iter().cloned()))
                .build();
            self.write_method_resolution(tgt_expr, func, subst);
        }
    }

    fn infer_expr_array(
        &mut self,
        array: &Array,
        expected: &Expectation,
    ) -> chalk_ir::Ty<Interner> {
        let elem_ty = match expected.to_option(&mut self.table).as_ref().map(|t| t.kind(Interner)) {
            Some(TyKind::Array(st, _) | TyKind::Slice(st)) => st.clone(),
            _ => self.table.new_type_var(),
        };

        let krate = self.resolver.krate();

        let expected = Expectation::has_type(elem_ty.clone());
        let (elem_ty, len) = match array {
            Array::ElementList { elements, .. } if elements.is_empty() => {
                (elem_ty, consteval::usize_const(self.db, Some(0), krate))
            }
            Array::ElementList { elements, .. } => {
                let mut coerce = CoerceMany::new(elem_ty);
                for &expr in elements.iter() {
                    let cur_elem_ty = self.infer_expr_inner(expr, &expected, ExprIsRead::Yes);
                    coerce.coerce(self, Some(expr), &cur_elem_ty, CoercionCause::Expr(expr));
                }
                (
                    coerce.complete(self),
                    consteval::usize_const(self.db, Some(elements.len() as u128), krate),
                )
            }
            &Array::Repeat { initializer, repeat } => {
                self.infer_expr_coerce(
                    initializer,
                    &Expectation::has_type(elem_ty.clone()),
                    ExprIsRead::Yes,
                );
                let usize = TyKind::Scalar(Scalar::Uint(UintTy::Usize)).intern(Interner);
                match self.body[repeat] {
                    Expr::Underscore => {
                        self.write_expr_ty(repeat, usize);
                    }
                    _ => _ = self.infer_expr(repeat, &Expectation::HasType(usize), ExprIsRead::Yes),
                }

                (
                    elem_ty,
                    consteval::eval_to_const(
                        repeat,
                        ParamLoweringMode::Placeholder,
                        self,
                        DebruijnIndex::INNERMOST,
                    ),
                )
            }
        };
        // Try to evaluate unevaluated constant, and insert variable if is not possible.
        let len = self.table.insert_const_vars_shallow(len);
        TyKind::Array(elem_ty, len).intern(Interner)
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
        coerce_many.coerce(self, Some(expr), &return_expr_ty, CoercionCause::Expr(expr));
        self.return_coercion = Some(coerce_many);
    }

    fn infer_expr_return(&mut self, ret: ExprId, expr: Option<ExprId>) -> Ty {
        match self.return_coercion {
            Some(_) => {
                if let Some(expr) = expr {
                    self.infer_return(expr);
                } else {
                    let mut coerce = self.return_coercion.take().unwrap();
                    coerce.coerce_forced_unit(self, CoercionCause::Expr(ret));
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
        self.result.standard_types.never.clone()
    }

    fn infer_expr_become(&mut self, expr: ExprId) -> Ty {
        match &self.return_coercion {
            Some(return_coercion) => {
                let ret_ty = return_coercion.expected_ty();

                let call_expr_ty = self.infer_expr_inner(
                    expr,
                    &Expectation::HasType(ret_ty.clone()),
                    ExprIsRead::Yes,
                );

                // NB: this should *not* coerce.
                //     tail calls don't support any coercions except lifetimes ones (like `&'static u8 -> &'a u8`).
                self.unify(&call_expr_ty, &ret_ty);
            }
            None => {
                // FIXME: diagnose `become` outside of functions
                self.infer_expr_no_expect(expr, ExprIsRead::Yes);
            }
        }

        self.result.standard_types.never.clone()
    }

    fn infer_expr_box(&mut self, inner_expr: ExprId, expected: &Expectation) -> Ty {
        if let Some(box_id) = self.resolve_boxed_box() {
            let table = &mut self.table;
            let inner_exp = expected
                .to_option(table)
                .as_ref()
                .and_then(|e| e.as_adt())
                .filter(|(e_adt, _)| e_adt == &box_id)
                .map(|(_, subts)| {
                    let g = subts.at(Interner, 0);
                    Expectation::rvalue_hint(self, Ty::clone(g.assert_ty_ref(Interner)))
                })
                .unwrap_or_else(Expectation::none);

            let inner_ty = self.infer_expr_inner(inner_expr, &inner_exp, ExprIsRead::Yes);
            TyBuilder::adt(self.db, box_id)
                .push(inner_ty)
                .fill_with_defaults(self.db, || self.table.new_type_var())
                .build()
        } else {
            self.err_ty()
        }
    }

    fn infer_overloadable_binop(
        &mut self,
        lhs: ExprId,
        op: BinaryOp,
        rhs: ExprId,
        tgt_expr: ExprId,
    ) -> Ty {
        let lhs_expectation = Expectation::none();
        let is_read = if matches!(op, BinaryOp::Assignment { .. }) {
            ExprIsRead::Yes
        } else {
            ExprIsRead::No
        };
        let lhs_ty = self.infer_expr(lhs, &lhs_expectation, is_read);
        let rhs_ty = self.table.new_type_var();

        let trait_func = lang_items_for_bin_op(op).and_then(|(name, lang_item)| {
            let trait_id = self.resolve_lang_item(lang_item)?.as_trait()?;
            let func = trait_id.trait_items(self.db).method_by_name(&name)?;
            Some((trait_id, func))
        });
        let (trait_, func) = match trait_func {
            Some(it) => it,
            None => {
                // HACK: `rhs_ty` is a general inference variable with no clue at all at this
                // point. Passing `lhs_ty` as both operands just to check if `lhs_ty` is a builtin
                // type applicable to `op`.
                let ret_ty = if self.is_builtin_binop(&lhs_ty, &lhs_ty, op) {
                    // Assume both operands are builtin so we can continue inference. No guarantee
                    // on the correctness, rustc would complain as necessary lang items don't seem
                    // to exist anyway.
                    self.enforce_builtin_binop_types(&lhs_ty, &rhs_ty, op)
                } else {
                    self.err_ty()
                };

                self.infer_expr_coerce(rhs, &Expectation::has_type(rhs_ty), ExprIsRead::Yes);

                return ret_ty;
            }
        };

        // HACK: We can use this substitution for the function because the function itself doesn't
        // have its own generic parameters.
        let subst = TyBuilder::subst_for_def(self.db, trait_, None);
        if subst.remaining() != 2 {
            return Ty::new(Interner, TyKind::Error);
        }
        let subst = subst.push(lhs_ty.clone()).push(rhs_ty.clone()).build();

        self.write_method_resolution(tgt_expr, func, subst.clone());

        let method_ty = self.db.value_ty(func.into()).unwrap().substitute(Interner, &subst);
        self.register_obligations_for_call(&method_ty);

        self.infer_expr_coerce(rhs, &Expectation::has_type(rhs_ty.clone()), ExprIsRead::Yes);

        let ret_ty = match method_ty.callable_sig(self.db) {
            Some(sig) => {
                let p_left = &sig.params()[0];
                if matches!(op, BinaryOp::CmpOp(..) | BinaryOp::Assignment { .. }) {
                    if let TyKind::Ref(mtbl, lt, _) = p_left.kind(Interner) {
                        self.write_expr_adj(
                            lhs,
                            Box::new([Adjustment {
                                kind: Adjust::Borrow(AutoBorrow::Ref(lt.clone(), *mtbl)),
                                target: p_left.clone(),
                            }]),
                        );
                    }
                }
                let p_right = &sig.params()[1];
                if matches!(op, BinaryOp::CmpOp(..)) {
                    if let TyKind::Ref(mtbl, lt, _) = p_right.kind(Interner) {
                        self.write_expr_adj(
                            rhs,
                            Box::new([Adjustment {
                                kind: Adjust::Borrow(AutoBorrow::Ref(lt.clone(), *mtbl)),
                                target: p_right.clone(),
                            }]),
                        );
                    }
                }
                sig.ret().clone()
            }
            None => self.err_ty(),
        };

        let ret_ty = self.normalize_associated_types_in(ret_ty);

        if self.is_builtin_binop(&lhs_ty, &rhs_ty, op) {
            // use knowledge of built-in binary ops, which can sometimes help inference
            let builtin_ret = self.enforce_builtin_binop_types(&lhs_ty, &rhs_ty, op);
            self.unify(&builtin_ret, &ret_ty);
        }

        ret_ty
    }

    fn infer_block(
        &mut self,
        expr: ExprId,
        block_id: Option<BlockId>,
        statements: &[Statement],
        tail: Option<ExprId>,
        label: Option<LabelId>,
        expected: &Expectation,
    ) -> Ty {
        let coerce_ty = expected.coercion_target_type(&mut self.table);
        let g = self.resolver.update_to_inner_scope(self.db, self.owner, expr);
        let prev_env = block_id.map(|block_id| {
            let prev_env = self.table.trait_env.clone();
            TraitEnvironment::with_block(&mut self.table.trait_env, block_id);
            prev_env
        });

        let (break_ty, ty) =
            self.with_breakable_ctx(BreakableKind::Block, Some(coerce_ty), label, |this| {
                for stmt in statements {
                    match stmt {
                        Statement::Let { pat, type_ref, initializer, else_branch } => {
                            let decl_ty = type_ref
                                .as_ref()
                                .map(|&tr| this.make_body_ty(tr))
                                .unwrap_or_else(|| this.table.new_type_var());

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
                                        &Expectation::has_type(decl_ty.clone()),
                                        target_is_read,
                                    )
                                } else {
                                    this.infer_expr_coerce(
                                        *expr,
                                        &Expectation::has_type(decl_ty.clone()),
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

                            this.infer_top_pat(*pat, &ty, Some(decl));
                            if let Some(expr) = else_branch {
                                let previous_diverges =
                                    mem::replace(&mut this.diverges, Diverges::Maybe);
                                this.infer_expr_coerce(
                                    *expr,
                                    &Expectation::HasType(this.result.standard_types.never.clone()),
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
                                    &Expectation::HasType(this.result.standard_types.unit.clone()),
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
                        let coerce_never = if this
                            .expr_guaranteed_to_constitute_read_for_never(expr, ExprIsRead::Yes)
                        {
                            CoerceNever::Yes
                        } else {
                            CoerceNever::No
                        };
                        if this
                            .coerce(
                                Some(expr),
                                &this.result.standard_types.unit.clone(),
                                &t,
                                coerce_never,
                            )
                            .is_err()
                        {
                            this.result.type_mismatches.insert(
                                expr.into(),
                                TypeMismatch {
                                    expected: t.clone(),
                                    actual: this.result.standard_types.unit.clone(),
                                },
                            );
                        }
                        t
                    } else {
                        this.result.standard_types.unit.clone()
                    }
                }
            });
        self.resolver.reset_to_guard(g);
        if let Some(prev_env) = prev_env {
            self.table.trait_env = prev_env;
        }

        break_ty.unwrap_or(ty)
    }

    fn lookup_field(
        &mut self,
        receiver_ty: &Ty,
        name: &Name,
    ) -> Option<(Ty, Either<FieldId, TupleFieldId>, Vec<Adjustment>, bool)> {
        let mut autoderef = Autoderef::new(&mut self.table, receiver_ty.clone(), false, false);
        let mut private_field = None;
        let res = autoderef.by_ref().find_map(|(derefed_ty, _)| {
            let (field_id, parameters) = match derefed_ty.kind(Interner) {
                TyKind::Tuple(_, substs) => {
                    return name.as_tuple_index().and_then(|idx| {
                        substs
                            .as_slice(Interner)
                            .get(idx)
                            .map(|a| a.assert_ty_ref(Interner))
                            .cloned()
                            .map(|ty| {
                                (
                                    Either::Right(TupleFieldId {
                                        tuple: TupleId(
                                            self.tuple_field_accesses_rev
                                                .insert_full(substs.clone())
                                                .0
                                                as u32,
                                        ),
                                        index: idx as u32,
                                    }),
                                    ty,
                                )
                            })
                    });
                }
                &TyKind::Adt(AdtId(hir_def::AdtId::StructId(s)), ref parameters) => {
                    let local_id = s.fields(self.db).field(name)?;
                    let field = FieldId { parent: s.into(), local_id };
                    (field, parameters.clone())
                }
                &TyKind::Adt(AdtId(hir_def::AdtId::UnionId(u)), ref parameters) => {
                    let local_id = u.fields(self.db).field(name)?;
                    let field = FieldId { parent: u.into(), local_id };
                    (field, parameters.clone())
                }
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
                .clone()
                .substitute(Interner, &parameters);
            Some((Either::Left(field_id), ty))
        });

        Some(match res {
            Some((field_id, ty)) => {
                let adjustments = auto_deref_adjust_steps(&autoderef);
                let ty = self.insert_type_vars(ty);
                let ty = self.normalize_associated_types_in(ty);

                (ty, field_id, adjustments, true)
            }
            None => {
                let (field_id, subst) = private_field?;
                let adjustments = auto_deref_adjust_steps(&autoderef);
                let ty = self.db.field_types(field_id.parent)[field_id.local_id]
                    .clone()
                    .substitute(Interner, &subst);
                let ty = self.insert_type_vars(ty);
                let ty = self.normalize_associated_types_in(ty);

                (ty, Either::Left(field_id), adjustments, false)
            }
        })
    }

    fn infer_field_access(
        &mut self,
        tgt_expr: ExprId,
        receiver: ExprId,
        name: &Name,
        expected: &Expectation,
    ) -> Ty {
        // Field projections don't constitute reads.
        let receiver_ty = self.infer_expr_inner(receiver, &Expectation::none(), ExprIsRead::No);

        if name.is_missing() {
            // Bail out early, don't even try to look up field. Also, we don't issue an unresolved
            // field diagnostic because this is a syntax error rather than a semantic error.
            return self.err_ty();
        }

        match self.lookup_field(&receiver_ty, name) {
            Some((ty, field_id, adjustments, is_public)) => {
                self.write_expr_adj(receiver, adjustments.into_boxed_slice());
                self.result.field_resolutions.insert(tgt_expr, field_id);
                if !is_public {
                    if let Either::Left(field) = field_id {
                        // FIXME: Merge this diagnostic into UnresolvedField?
                        self.push_diagnostic(InferenceDiagnostic::PrivateField {
                            expr: tgt_expr,
                            field,
                        });
                    }
                }
                ty
            }
            None => {
                // no field found, lets attempt to resolve it like a function so that IDE things
                // work out while people are typing
                let canonicalized_receiver = self.canonicalize(receiver_ty.clone());
                let resolved = method_resolution::lookup_method(
                    self.db,
                    &canonicalized_receiver,
                    self.table.trait_env.clone(),
                    self.get_traits_in_scope().as_ref().left_or_else(|&it| it),
                    VisibleFromModule::Filter(self.resolver.module()),
                    name,
                );
                self.push_diagnostic(InferenceDiagnostic::UnresolvedField {
                    expr: tgt_expr,
                    receiver: receiver_ty.clone(),
                    name: name.clone(),
                    method_with_same_name_exists: resolved.is_some(),
                });
                match resolved {
                    Some((adjust, func, _)) => {
                        let (ty, adjustments) = adjust.apply(&mut self.table, receiver_ty);
                        let substs = self.substs_for_method_call(tgt_expr, func.into(), None);
                        self.write_expr_adj(receiver, adjustments.into_boxed_slice());
                        self.write_method_resolution(tgt_expr, func, substs.clone());

                        self.check_method_call(
                            tgt_expr,
                            &[],
                            self.db.value_ty(func.into()).unwrap(),
                            substs,
                            ty,
                            expected,
                        )
                    }
                    None => self.err_ty(),
                }
            }
        }
    }

    fn infer_call(
        &mut self,
        tgt_expr: ExprId,
        callee: ExprId,
        args: &[ExprId],
        expected: &Expectation,
    ) -> Ty {
        let callee_ty = self.infer_expr(callee, &Expectation::none(), ExprIsRead::Yes);
        let mut derefs = Autoderef::new(&mut self.table, callee_ty.clone(), false, true);
        let (res, derefed_callee) = loop {
            let Some((callee_deref_ty, _)) = derefs.next() else {
                break (None, callee_ty.clone());
            };
            if let Some(res) = derefs.table.callable_sig(&callee_deref_ty, args.len()) {
                break (Some(res), callee_deref_ty);
            }
        };
        // if the function is unresolved, we use is_varargs=true to
        // suppress the arg count diagnostic here
        let is_varargs =
            derefed_callee.callable_sig(self.db).is_some_and(|sig| sig.is_varargs) || res.is_none();
        let (param_tys, ret_ty) = match res {
            Some((func, params, ret_ty)) => {
                let mut adjustments = auto_deref_adjust_steps(&derefs);
                if let TyKind::Closure(c, _) =
                    self.table.resolve_completely(callee_ty.clone()).kind(Interner)
                {
                    self.add_current_closure_dependency(*c);
                    self.deferred_closures.entry(*c).or_default().push((
                        derefed_callee.clone(),
                        callee_ty.clone(),
                        params.clone(),
                        tgt_expr,
                    ));
                }
                if let Some(fn_x) = func {
                    self.write_fn_trait_method_resolution(
                        fn_x,
                        &derefed_callee,
                        &mut adjustments,
                        &callee_ty,
                        &params,
                        tgt_expr,
                    );
                }
                self.write_expr_adj(callee, adjustments.into_boxed_slice());
                (params, ret_ty)
            }
            None => {
                self.push_diagnostic(InferenceDiagnostic::ExpectedFunction {
                    call_expr: tgt_expr,
                    found: callee_ty.clone(),
                });
                (Vec::new(), self.err_ty())
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
        callee_ty: Ty,
        param_tys: &[Ty],
        ret_ty: Ty,
        indices_to_skip: &[u32],
        is_varargs: bool,
        expected: &Expectation,
    ) -> Ty {
        self.register_obligations_for_call(&callee_ty);

        let expected_inputs = self.expected_inputs_for_expected_output(
            expected,
            ret_ty.clone(),
            param_tys.to_owned(),
        );

        self.check_call_arguments(
            tgt_expr,
            args,
            &expected_inputs,
            param_tys,
            indices_to_skip,
            is_varargs,
        );
        self.normalize_associated_types_in(ret_ty)
    }

    fn infer_method_call(
        &mut self,
        tgt_expr: ExprId,
        receiver: ExprId,
        args: &[ExprId],
        method_name: &Name,
        generic_args: Option<&GenericArgs>,
        expected: &Expectation,
    ) -> Ty {
        let receiver_ty = self.infer_expr_inner(receiver, &Expectation::none(), ExprIsRead::Yes);
        let canonicalized_receiver = self.canonicalize(receiver_ty.clone());

        let resolved = method_resolution::lookup_method(
            self.db,
            &canonicalized_receiver,
            self.table.trait_env.clone(),
            self.get_traits_in_scope().as_ref().left_or_else(|&it| it),
            VisibleFromModule::Filter(self.resolver.module()),
            method_name,
        );
        match resolved {
            Some((adjust, func, visible)) => {
                if !visible {
                    self.push_diagnostic(InferenceDiagnostic::PrivateAssocItem {
                        id: tgt_expr.into(),
                        item: func.into(),
                    })
                }

                let (ty, adjustments) = adjust.apply(&mut self.table, receiver_ty);
                self.write_expr_adj(receiver, adjustments.into_boxed_slice());

                let substs = self.substs_for_method_call(tgt_expr, func.into(), generic_args);
                self.write_method_resolution(tgt_expr, func, substs.clone());
                self.check_method_call(
                    tgt_expr,
                    args,
                    self.db.value_ty(func.into()).expect("we have a function def"),
                    substs,
                    ty,
                    expected,
                )
            }
            // Failed to resolve, report diagnostic and try to resolve as call to field access or
            // assoc function
            None => {
                let field_with_same_name_exists = match self.lookup_field(&receiver_ty, method_name)
                {
                    Some((ty, field_id, adjustments, _public)) => {
                        self.write_expr_adj(receiver, adjustments.into_boxed_slice());
                        self.result.field_resolutions.insert(tgt_expr, field_id);
                        Some(ty)
                    }
                    None => None,
                };

                let assoc_func_with_same_name = method_resolution::iterate_method_candidates(
                    &canonicalized_receiver,
                    self.db,
                    self.table.trait_env.clone(),
                    self.get_traits_in_scope().as_ref().left_or_else(|&it| it),
                    VisibleFromModule::Filter(self.resolver.module()),
                    Some(method_name),
                    method_resolution::LookupMode::Path,
                    |_ty, item, visible| match item {
                        hir_def::AssocItemId::FunctionId(function_id) if visible => {
                            Some(function_id)
                        }
                        _ => None,
                    },
                );

                self.push_diagnostic(InferenceDiagnostic::UnresolvedMethodCall {
                    expr: tgt_expr,
                    receiver: receiver_ty.clone(),
                    name: method_name.clone(),
                    field_with_same_name: field_with_same_name_exists.clone(),
                    assoc_func_with_same_name,
                });

                let recovered = match assoc_func_with_same_name {
                    Some(f) => {
                        let substs = self.substs_for_method_call(tgt_expr, f.into(), generic_args);
                        let f = self
                            .db
                            .value_ty(f.into())
                            .expect("we have a function def")
                            .substitute(Interner, &substs);
                        let sig = f.callable_sig(self.db).expect("we have a function def");
                        Some((f, sig, true))
                    }
                    None => field_with_same_name_exists.and_then(|field_ty| {
                        let callable_sig = field_ty.callable_sig(self.db)?;
                        Some((field_ty, callable_sig, false))
                    }),
                };
                match recovered {
                    Some((callee_ty, sig, strip_first)) => self.check_call(
                        tgt_expr,
                        args,
                        callee_ty,
                        sig.params().get(strip_first as usize..).unwrap_or(&[]),
                        sig.ret().clone(),
                        &[],
                        true,
                        expected,
                    ),
                    None => {
                        self.check_call_arguments(tgt_expr, args, &[], &[], &[], true);
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
        method_ty: Binders<Ty>,
        substs: Substitution,
        receiver_ty: Ty,
        expected: &Expectation,
    ) -> Ty {
        let method_ty = method_ty.substitute(Interner, &substs);
        self.register_obligations_for_call(&method_ty);
        let ((formal_receiver_ty, param_tys), ret_ty, is_varargs) =
            match method_ty.callable_sig(self.db) {
                Some(sig) => (
                    if !sig.params().is_empty() {
                        (sig.params()[0].clone(), sig.params()[1..].to_vec())
                    } else {
                        (self.err_ty(), Vec::new())
                    },
                    sig.ret().clone(),
                    sig.is_varargs,
                ),
                None => ((self.err_ty(), Vec::new()), self.err_ty(), true),
            };
        self.unify(&formal_receiver_ty, &receiver_ty);

        let expected_inputs =
            self.expected_inputs_for_expected_output(expected, ret_ty.clone(), param_tys.clone());

        self.check_call_arguments(tgt_expr, args, &expected_inputs, &param_tys, &[], is_varargs);
        self.normalize_associated_types_in(ret_ty)
    }

    fn expected_inputs_for_expected_output(
        &mut self,
        expected_output: &Expectation,
        output: Ty,
        inputs: Vec<Ty>,
    ) -> Vec<Ty> {
        if let Some(expected_ty) = expected_output.only_has_type(&mut self.table) {
            self.table.fudge_inference(|table| {
                if table.try_unify(&expected_ty, &output).is_ok() {
                    table.resolve_with_fallback(inputs, &|var, kind, _, _| match kind {
                        chalk_ir::VariableKind::Ty(tk) => var.to_ty(Interner, tk).cast(Interner),
                        chalk_ir::VariableKind::Lifetime => {
                            var.to_lifetime(Interner).cast(Interner)
                        }
                        chalk_ir::VariableKind::Const(ty) => {
                            var.to_const(Interner, ty).cast(Interner)
                        }
                    })
                } else {
                    Vec::new()
                }
            })
        } else {
            Vec::new()
        }
    }

    fn check_call_arguments(
        &mut self,
        expr: ExprId,
        args: &[ExprId],
        expected_inputs: &[Ty],
        param_tys: &[Ty],
        skip_indices: &[u32],
        ignore_arg_param_mismatch: bool,
    ) {
        let arg_count_mismatch =
            !ignore_arg_param_mismatch && args.len() != param_tys.len() + skip_indices.len();
        if arg_count_mismatch {
            self.push_diagnostic(InferenceDiagnostic::MismatchedArgCount {
                call_expr: expr,
                expected: param_tys.len() + skip_indices.len(),
                found: args.len(),
            });
        };

        // Quoting https://github.com/rust-lang/rust/blob/6ef275e6c3cb1384ec78128eceeb4963ff788dca/src/librustc_typeck/check/mod.rs#L3325 --
        // We do this in a pretty awful way: first we type-check any arguments
        // that are not closures, then we type-check the closures. This is so
        // that we have more information about the types of arguments when we
        // type-check the functions. This isn't really the right way to do this.
        for check_closures in [false, true] {
            let mut skip_indices = skip_indices.iter().copied().fuse().peekable();
            let param_iter = param_tys.iter().cloned().chain(repeat(self.err_ty()));
            let expected_iter = expected_inputs
                .iter()
                .cloned()
                .chain(param_iter.clone().skip(expected_inputs.len()));
            for (idx, ((&arg, param_ty), expected_ty)) in
                args.iter().zip(param_iter).zip(expected_iter).enumerate()
            {
                let is_closure = matches!(&self.body[arg], Expr::Closure { .. });
                if is_closure != check_closures {
                    continue;
                }

                while skip_indices.peek().is_some_and(|&i| i < idx as u32) {
                    skip_indices.next();
                }
                if skip_indices.peek().copied() == Some(idx as u32) {
                    continue;
                }

                // the difference between param_ty and expected here is that
                // expected is the parameter when the expected *return* type is
                // taken into account. So in `let _: &[i32] = identity(&[1, 2])`
                // the expected type is already `&[i32]`, whereas param_ty is
                // still an unbound type variable. We don't always want to force
                // the parameter to coerce to the expected type (for example in
                // `coerce_unsize_expected_type_4`).
                let param_ty = self.normalize_associated_types_in(param_ty);
                let expected_ty = self.normalize_associated_types_in(expected_ty);
                let expected = Expectation::rvalue_hint(self, expected_ty);
                // infer with the expected type we have...
                let ty = self.infer_expr_inner(arg, &expected, ExprIsRead::Yes);

                // then coerce to either the expected type or just the formal parameter type
                let coercion_target = if let Some(ty) = expected.only_has_type(&mut self.table) {
                    // if we are coercing to the expectation, unify with the
                    // formal parameter type to connect everything
                    self.unify(&ty, &param_ty);
                    ty
                } else {
                    param_ty
                };
                // The function signature may contain some unknown types, so we need to insert
                // type vars here to avoid type mismatch false positive.
                let coercion_target = self.insert_type_vars(coercion_target);

                // Any expression that produces a value of type `!` must have diverged,
                // unless it's a place expression that isn't being read from, in which case
                // diverging would be unsound since we may never actually read the `!`.
                // e.g. `let _ = *never_ptr;` with `never_ptr: *const !`.
                let coerce_never =
                    if self.expr_guaranteed_to_constitute_read_for_never(arg, ExprIsRead::Yes) {
                        CoerceNever::Yes
                    } else {
                        CoerceNever::No
                    };
                if self.coerce(Some(arg), &ty, &coercion_target, coerce_never).is_err()
                    && !arg_count_mismatch
                {
                    self.result.type_mismatches.insert(
                        arg.into(),
                        TypeMismatch { expected: coercion_target, actual: ty.clone() },
                    );
                }
            }
        }
    }

    fn substs_for_method_call(
        &mut self,
        expr: ExprId,
        def: GenericDefId,
        generic_args: Option<&GenericArgs>,
    ) -> Substitution {
        struct LowererCtx<'a, 'b> {
            ctx: &'a mut InferenceContext<'b>,
            expr: ExprId,
        }

        impl GenericArgsLowerer for LowererCtx<'_, '_> {
            fn report_len_mismatch(
                &mut self,
                def: GenericDefId,
                provided_count: u32,
                expected_count: u32,
                kind: IncorrectGenericsLenKind,
            ) {
                self.ctx.push_diagnostic(InferenceDiagnostic::MethodCallIncorrectGenericsLen {
                    expr: self.expr,
                    provided_count,
                    expected_count,
                    kind,
                    def,
                });
            }

            fn report_arg_mismatch(
                &mut self,
                param_id: GenericParamId,
                arg_idx: u32,
                has_self_arg: bool,
            ) {
                self.ctx.push_diagnostic(InferenceDiagnostic::MethodCallIncorrectGenericsOrder {
                    expr: self.expr,
                    param_id,
                    arg_idx,
                    has_self_arg,
                });
            }

            fn provided_kind(
                &mut self,
                param_id: GenericParamId,
                param: GenericParamDataRef<'_>,
                arg: &GenericArg,
            ) -> crate::GenericArg {
                match (param, arg) {
                    (GenericParamDataRef::LifetimeParamData(_), GenericArg::Lifetime(lifetime)) => {
                        self.ctx.make_body_lifetime(*lifetime).cast(Interner)
                    }
                    (GenericParamDataRef::TypeParamData(_), GenericArg::Type(type_ref)) => {
                        self.ctx.make_body_ty(*type_ref).cast(Interner)
                    }
                    (GenericParamDataRef::ConstParamData(_), GenericArg::Const(konst)) => {
                        let GenericParamId::ConstParamId(const_id) = param_id else {
                            unreachable!("non-const param ID for const param");
                        };
                        let const_ty = self.ctx.db.const_param_ty(const_id);
                        self.ctx.make_body_const(*konst, const_ty).cast(Interner)
                    }
                    _ => unreachable!("unmatching param kinds were passed to `provided_kind()`"),
                }
            }

            fn provided_type_like_const(
                &mut self,
                const_ty: Ty,
                arg: TypeLikeConst<'_>,
            ) -> crate::Const {
                match arg {
                    TypeLikeConst::Path(path) => self.ctx.make_path_as_body_const(path, const_ty),
                    TypeLikeConst::Infer => self.ctx.table.new_const_var(const_ty),
                }
            }

            fn inferred_kind(
                &mut self,
                _def: GenericDefId,
                param_id: GenericParamId,
                _param: GenericParamDataRef<'_>,
                _infer_args: bool,
                _preceding_args: &[crate::GenericArg],
            ) -> crate::GenericArg {
                // Always create an inference var, even when `infer_args == false`. This helps with diagnostics,
                // and I think it's also required in the presence of `impl Trait` (that must be inferred).
                match param_id {
                    GenericParamId::TypeParamId(_) => self.ctx.table.new_type_var().cast(Interner),
                    GenericParamId::ConstParamId(const_id) => self
                        .ctx
                        .table
                        .new_const_var(self.ctx.db.const_param_ty(const_id))
                        .cast(Interner),
                    GenericParamId::LifetimeParamId(_) => {
                        self.ctx.table.new_lifetime_var().cast(Interner)
                    }
                }
            }

            fn parent_arg(&mut self, param_id: GenericParamId) -> crate::GenericArg {
                match param_id {
                    GenericParamId::TypeParamId(_) => self.ctx.table.new_type_var().cast(Interner),
                    GenericParamId::ConstParamId(const_id) => self
                        .ctx
                        .table
                        .new_const_var(self.ctx.db.const_param_ty(const_id))
                        .cast(Interner),
                    GenericParamId::LifetimeParamId(_) => {
                        self.ctx.table.new_lifetime_var().cast(Interner)
                    }
                }
            }

            fn report_elided_lifetimes_in_path(
                &mut self,
                _def: GenericDefId,
                _expected_count: u32,
                _hard_error: bool,
            ) {
                unreachable!("we set `LifetimeElisionKind::Infer`")
            }

            fn report_elision_failure(&mut self, _def: GenericDefId, _expected_count: u32) {
                unreachable!("we set `LifetimeElisionKind::Infer`")
            }

            fn report_missing_lifetime(&mut self, _def: GenericDefId, _expected_count: u32) {
                unreachable!("we set `LifetimeElisionKind::Infer`")
            }
        }

        substs_from_args_and_bindings(
            self.db,
            self.body,
            generic_args,
            def,
            true,
            LifetimeElisionKind::Infer,
            false,
            None,
            &mut LowererCtx { ctx: self, expr },
        )
    }

    fn register_obligations_for_call(&mut self, callable_ty: &Ty) {
        let callable_ty = self.resolve_ty_shallow(callable_ty);
        if let TyKind::FnDef(fn_def, parameters) = callable_ty.kind(Interner) {
            let def: CallableDefId = from_chalk(self.db, *fn_def);
            let generic_predicates =
                self.db.generic_predicates(GenericDefId::from_callable(self.db, def));
            for predicate in generic_predicates.iter() {
                let (predicate, binders) = predicate
                    .clone()
                    .substitute(Interner, parameters)
                    .into_value_and_skipped_binders();
                always!(binders.len(Interner) == 0); // quantified where clauses not yet handled
                self.push_obligation(predicate.cast(Interner));
            }
            // add obligation for trait implementation, if this is a trait method
            match def {
                CallableDefId::FunctionId(f) => {
                    if let ItemContainerId::TraitId(trait_) = f.lookup(self.db).container {
                        // construct a TraitRef
                        let trait_params_len = generics(self.db, trait_.into()).len();
                        let substs = Substitution::from_iter(
                            Interner,
                            // The generic parameters for the trait come after those for the
                            // function.
                            &parameters.as_slice(Interner)[..trait_params_len],
                        );
                        self.push_obligation(
                            TraitRef { trait_id: to_chalk_trait_id(trait_), substitution: substs }
                                .cast(Interner),
                        );
                    }
                }
                CallableDefId::StructId(_) | CallableDefId::EnumVariantId(_) => {}
            }
        }
    }

    /// Returns the argument indices to skip.
    fn check_legacy_const_generics(&mut self, callee: Ty, args: &[ExprId]) -> Box<[u32]> {
        let (func, subst) = match callee.kind(Interner) {
            TyKind::FnDef(fn_id, subst) => {
                let callable = CallableDefId::from_chalk(self.db, *fn_id);
                let func = match callable {
                    CallableDefId::FunctionId(f) => f,
                    _ => return Default::default(),
                };
                (func, subst)
            }
            _ => return Default::default(),
        };

        let data = self.db.function_signature(func);
        let Some(legacy_const_generics_indices) = &data.legacy_const_generics_indices else {
            return Default::default();
        };

        // only use legacy const generics if the param count matches with them
        if data.params.len() + legacy_const_generics_indices.len() != args.len() {
            if args.len() <= data.params.len() {
                return Default::default();
            } else {
                // there are more parameters than there should be without legacy
                // const params; use them
                let mut indices = legacy_const_generics_indices.as_ref().clone();
                indices.sort();
                return indices;
            }
        }

        // check legacy const parameters
        for (subst_idx, arg_idx) in legacy_const_generics_indices.iter().copied().enumerate() {
            let arg = match subst.at(Interner, subst_idx).constant(Interner) {
                Some(c) => c,
                None => continue, // not a const parameter?
            };
            if arg_idx >= args.len() as u32 {
                continue;
            }
            let _ty = arg.data(Interner).ty.clone();
            let expected = Expectation::none(); // FIXME use actual const ty, when that is lowered correctly
            self.infer_expr(args[arg_idx as usize], &expected, ExprIsRead::Yes);
            // FIXME: evaluate and unify with the const
        }
        let mut indices = legacy_const_generics_indices.as_ref().clone();
        indices.sort();
        indices
    }

    /// Dereferences a single level of immutable referencing.
    fn deref_ty_if_possible(&mut self, ty: &Ty) -> Ty {
        let ty = self.resolve_ty_shallow(ty);
        match ty.kind(Interner) {
            TyKind::Ref(Mutability::Not, _, inner) => self.resolve_ty_shallow(inner),
            _ => ty,
        }
    }

    /// Enforces expectations on lhs type and rhs type depending on the operator and returns the
    /// output type of the binary op.
    fn enforce_builtin_binop_types(&mut self, lhs: &Ty, rhs: &Ty, op: BinaryOp) -> Ty {
        // Special-case a single layer of referencing, so that things like `5.0 + &6.0f32` work (See rust-lang/rust#57447).
        let lhs = self.deref_ty_if_possible(lhs);
        let rhs = self.deref_ty_if_possible(rhs);

        let (op, is_assign) = match op {
            BinaryOp::Assignment { op: Some(inner) } => (BinaryOp::ArithOp(inner), true),
            _ => (op, false),
        };

        let output_ty = match op {
            BinaryOp::LogicOp(_) => {
                let bool_ = self.result.standard_types.bool_.clone();
                self.unify(&lhs, &bool_);
                self.unify(&rhs, &bool_);
                bool_
            }

            BinaryOp::ArithOp(ArithOp::Shl | ArithOp::Shr) => {
                // result type is same as LHS always
                lhs
            }

            BinaryOp::ArithOp(_) => {
                // LHS, RHS, and result will have the same type
                self.unify(&lhs, &rhs);
                lhs
            }

            BinaryOp::CmpOp(_) => {
                // LHS and RHS will have the same type
                self.unify(&lhs, &rhs);
                self.result.standard_types.bool_.clone()
            }

            BinaryOp::Assignment { op: None } => {
                stdx::never!("Simple assignment operator is not binary op.");
                lhs
            }

            BinaryOp::Assignment { .. } => unreachable!("handled above"),
        };

        if is_assign { self.result.standard_types.unit.clone() } else { output_ty }
    }

    fn is_builtin_binop(&mut self, lhs: &Ty, rhs: &Ty, op: BinaryOp) -> bool {
        // Special-case a single layer of referencing, so that things like `5.0 + &6.0f32` work (See rust-lang/rust#57447).
        let lhs = self.deref_ty_if_possible(lhs);
        let rhs = self.deref_ty_if_possible(rhs);

        let op = match op {
            BinaryOp::Assignment { op: Some(inner) } => BinaryOp::ArithOp(inner),
            _ => op,
        };

        match op {
            BinaryOp::LogicOp(_) => true,

            BinaryOp::ArithOp(ArithOp::Shl | ArithOp::Shr) => {
                lhs.is_integral() && rhs.is_integral()
            }

            BinaryOp::ArithOp(
                ArithOp::Add | ArithOp::Sub | ArithOp::Mul | ArithOp::Div | ArithOp::Rem,
            ) => {
                lhs.is_integral() && rhs.is_integral()
                    || lhs.is_floating_point() && rhs.is_floating_point()
            }

            BinaryOp::ArithOp(ArithOp::BitAnd | ArithOp::BitOr | ArithOp::BitXor) => {
                lhs.is_integral() && rhs.is_integral()
                    || lhs.is_floating_point() && rhs.is_floating_point()
                    || matches!(
                        (lhs.kind(Interner), rhs.kind(Interner)),
                        (TyKind::Scalar(Scalar::Bool), TyKind::Scalar(Scalar::Bool))
                    )
            }

            BinaryOp::CmpOp(_) => {
                let is_scalar = |kind| {
                    matches!(
                        kind,
                        &TyKind::Scalar(_)
                            | TyKind::FnDef(..)
                            | TyKind::Function(_)
                            | TyKind::Raw(..)
                            | TyKind::InferenceVar(
                                _,
                                TyVariableKind::Integer | TyVariableKind::Float
                            )
                    )
                };
                is_scalar(lhs.kind(Interner)) && is_scalar(rhs.kind(Interner))
            }

            BinaryOp::Assignment { op: None } => {
                stdx::never!("Simple assignment operator is not binary op.");
                false
            }

            BinaryOp::Assignment { .. } => unreachable!("handled above"),
        }
    }

    pub(super) fn with_breakable_ctx<T>(
        &mut self,
        kind: BreakableKind,
        ty: Option<Ty>,
        label: Option<LabelId>,
        cb: impl FnOnce(&mut Self) -> T,
    ) -> (Option<Ty>, T) {
        self.breakables.push({
            BreakableContext { kind, may_break: false, coerce: ty.map(CoerceMany::new), label }
        });
        let res = cb(self);
        let ctx = self.breakables.pop().expect("breakable stack broken");
        (if ctx.may_break { ctx.coerce.map(|ctx| ctx.complete(self)) } else { None }, res)
    }
}
