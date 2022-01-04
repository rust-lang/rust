//! Type inference for expressions.

use std::{
    iter::{repeat, repeat_with},
    mem,
    sync::Arc,
};

use chalk_ir::{cast::Cast, fold::Shift, Mutability, TyVariableKind};
use hir_def::{
    expr::{
        ArithOp, Array, BinaryOp, CmpOp, Expr, ExprId, Literal, MatchGuard, Ordering, Statement,
        UnaryOp,
    },
    path::{GenericArg, GenericArgs},
    resolver::resolver_for_expr,
    FieldId, FunctionId, ItemContainerId, Lookup,
};
use hir_expand::name::{name, Name};
use stdx::always;
use syntax::ast::RangeOp;

use crate::{
    autoderef::{self, Autoderef},
    consteval,
    infer::coerce::CoerceMany,
    lower::lower_to_chalk_mutability,
    mapping::from_chalk,
    method_resolution,
    primitive::{self, UintTy},
    static_lifetime, to_chalk_trait_id,
    traits::FnTrait,
    utils::{generics, Generics},
    AdtId, Binders, CallableDefId, FnPointer, FnSig, FnSubst, InEnvironment, Interner,
    ProjectionTyExt, Rawness, Scalar, Substitution, TraitRef, Ty, TyBuilder, TyExt, TyKind,
};

use super::{
    find_breakable, BindingMode, BreakableContext, Diverges, Expectation, InferenceContext,
    InferenceDiagnostic, TypeMismatch,
};

impl<'a> InferenceContext<'a> {
    pub(super) fn infer_expr(&mut self, tgt_expr: ExprId, expected: &Expectation) -> Ty {
        let ty = self.infer_expr_inner(tgt_expr, expected);
        if self.resolve_ty_shallow(&ty).is_never() {
            // Any expression that produces a value of type `!` must have diverged
            self.diverges = Diverges::Always;
        }
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

    /// Infer type of expression with possibly implicit coerce to the expected type.
    /// Return the type after possible coercion.
    pub(super) fn infer_expr_coerce(&mut self, expr: ExprId, expected: &Expectation) -> Ty {
        let ty = self.infer_expr_inner(expr, expected);
        if let Some(target) = expected.only_has_type(&mut self.table) {
            match self.coerce(Some(expr), &ty, &target) {
                Ok(res) => res.value,
                Err(_) => {
                    self.result
                        .type_mismatches
                        .insert(expr.into(), TypeMismatch { expected: target, actual: ty.clone() });
                    // Return actual type when type mismatch.
                    // This is needed for diagnostic when return type mismatch.
                    ty
                }
            }
        } else {
            ty
        }
    }

    fn callable_sig_from_fn_trait(&mut self, ty: &Ty, num_args: usize) -> Option<(Vec<Ty>, Ty)> {
        let krate = self.resolver.krate()?;
        let fn_once_trait = FnTrait::FnOnce.get_id(self.db, krate)?;
        let output_assoc_type =
            self.db.trait_data(fn_once_trait).associated_type_by_name(&name![Output])?;

        let mut arg_tys = vec![];
        let arg_ty = TyBuilder::tuple(num_args)
            .fill(repeat_with(|| {
                let arg = self.table.new_type_var();
                arg_tys.push(arg.clone());
                arg
            }))
            .build();

        let projection = {
            let b = TyBuilder::assoc_type_projection(self.db, output_assoc_type);
            if b.remaining() != 2 {
                return None;
            }
            b.push(ty.clone()).push(arg_ty).build()
        };

        let trait_env = self.trait_env.env.clone();
        let obligation = InEnvironment {
            goal: projection.trait_ref(self.db).cast(Interner),
            environment: trait_env,
        };
        let canonical = self.canonicalize(obligation.clone());
        if self.db.trait_solve(krate, canonical.value.cast(Interner)).is_some() {
            self.push_obligation(obligation.goal);
            let return_ty = self.table.normalize_projection_ty(projection);
            Some((arg_tys, return_ty))
        } else {
            None
        }
    }

    pub(crate) fn callable_sig(&mut self, ty: &Ty, num_args: usize) -> Option<(Vec<Ty>, Ty)> {
        match ty.callable_sig(self.db) {
            Some(sig) => Some((sig.params().to_vec(), sig.ret().clone())),
            None => self.callable_sig_from_fn_trait(ty, num_args),
        }
    }

    fn infer_expr_inner(&mut self, tgt_expr: ExprId, expected: &Expectation) -> Ty {
        self.db.unwind_if_cancelled();

        let body = Arc::clone(&self.body); // avoid borrow checker problem
        let ty = match &body[tgt_expr] {
            Expr::Missing => self.err_ty(),
            &Expr::If { condition, then_branch, else_branch } => {
                // if let is desugared to match, so this is always simple if
                self.infer_expr(
                    condition,
                    &Expectation::has_type(TyKind::Scalar(Scalar::Bool).intern(Interner)),
                );

                let condition_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
                let mut both_arms_diverge = Diverges::Always;

                let result_ty = self.table.new_type_var();
                let then_ty = self.infer_expr_inner(then_branch, expected);
                both_arms_diverge &= mem::replace(&mut self.diverges, Diverges::Maybe);
                let mut coerce = CoerceMany::new(result_ty);
                coerce.coerce(self, Some(then_branch), &then_ty);
                let else_ty = match else_branch {
                    Some(else_branch) => self.infer_expr_inner(else_branch, expected),
                    None => TyBuilder::unit(),
                };
                both_arms_diverge &= self.diverges;
                // FIXME: create a synthetic `else {}` so we have something to refer to here instead of None?
                coerce.coerce(self, else_branch, &else_ty);

                self.diverges = condition_diverges | both_arms_diverge;

                coerce.complete()
            }
            Expr::Block { statements, tail, label, id: _ } => {
                let old_resolver = mem::replace(
                    &mut self.resolver,
                    resolver_for_expr(self.db.upcast(), self.owner, tgt_expr),
                );
                let ty = match label {
                    Some(_) => {
                        let break_ty = self.table.new_type_var();
                        self.breakables.push(BreakableContext {
                            may_break: false,
                            coerce: CoerceMany::new(break_ty.clone()),
                            label: label.map(|label| self.body[label].name.clone()),
                        });
                        let ty = self.infer_block(
                            tgt_expr,
                            statements,
                            *tail,
                            &Expectation::has_type(break_ty),
                        );
                        let ctxt = self.breakables.pop().expect("breakable stack broken");
                        if ctxt.may_break {
                            ctxt.coerce.complete()
                        } else {
                            ty
                        }
                    }
                    None => self.infer_block(tgt_expr, statements, *tail, expected),
                };
                self.resolver = old_resolver;
                ty
            }
            Expr::Unsafe { body } | Expr::Const { body } => self.infer_expr(*body, expected),
            Expr::TryBlock { body } => {
                let _inner = self.infer_expr(*body, expected);
                // FIXME should be std::result::Result<{inner}, _>
                self.err_ty()
            }
            Expr::Async { body } => {
                // Use the first type parameter as the output type of future.
                // existential type AsyncBlockImplTrait<InnerType>: Future<Output = InnerType>
                let inner_ty = self.infer_expr(*body, &Expectation::none());
                let impl_trait_id = crate::ImplTraitId::AsyncBlockTypeImplTrait(self.owner, *body);
                let opaque_ty_id = self.db.intern_impl_trait_id(impl_trait_id).into();
                TyKind::OpaqueType(opaque_ty_id, Substitution::from1(Interner, inner_ty))
                    .intern(Interner)
            }
            Expr::Loop { body, label } => {
                self.breakables.push(BreakableContext {
                    may_break: false,
                    coerce: CoerceMany::new(self.table.new_type_var()),
                    label: label.map(|label| self.body[label].name.clone()),
                });
                self.infer_expr(*body, &Expectation::has_type(TyBuilder::unit()));

                let ctxt = self.breakables.pop().expect("breakable stack broken");

                if ctxt.may_break {
                    self.diverges = Diverges::Maybe;
                    ctxt.coerce.complete()
                } else {
                    TyKind::Never.intern(Interner)
                }
            }
            Expr::While { condition, body, label } => {
                self.breakables.push(BreakableContext {
                    may_break: false,
                    coerce: CoerceMany::new(self.err_ty()),
                    label: label.map(|label| self.body[label].name.clone()),
                });
                // while let is desugared to a match loop, so this is always simple while
                self.infer_expr(
                    *condition,
                    &Expectation::has_type(TyKind::Scalar(Scalar::Bool).intern(Interner)),
                );
                self.infer_expr(*body, &Expectation::has_type(TyBuilder::unit()));
                let _ctxt = self.breakables.pop().expect("breakable stack broken");
                // the body may not run, so it diverging doesn't mean we diverge
                self.diverges = Diverges::Maybe;
                TyBuilder::unit()
            }
            Expr::For { iterable, body, pat, label } => {
                let iterable_ty = self.infer_expr(*iterable, &Expectation::none());

                self.breakables.push(BreakableContext {
                    may_break: false,
                    coerce: CoerceMany::new(self.err_ty()),
                    label: label.map(|label| self.body[label].name.clone()),
                });
                let pat_ty =
                    self.resolve_associated_type(iterable_ty, self.resolve_into_iter_item());

                self.infer_pat(*pat, &pat_ty, BindingMode::default());

                self.infer_expr(*body, &Expectation::has_type(TyBuilder::unit()));
                let _ctxt = self.breakables.pop().expect("breakable stack broken");
                // the body may not run, so it diverging doesn't mean we diverge
                self.diverges = Diverges::Maybe;
                TyBuilder::unit()
            }
            Expr::Lambda { body, args, ret_type, arg_types } => {
                assert_eq!(args.len(), arg_types.len());

                let mut sig_tys = Vec::new();

                // collect explicitly written argument types
                for arg_type in arg_types.iter() {
                    let arg_ty = match arg_type {
                        Some(type_ref) => self.make_ty(type_ref),
                        None => self.table.new_type_var(),
                    };
                    sig_tys.push(arg_ty);
                }

                // add return type
                let ret_ty = match ret_type {
                    Some(type_ref) => self.make_ty(type_ref),
                    None => self.table.new_type_var(),
                };
                sig_tys.push(ret_ty.clone());
                let sig_ty = TyKind::Function(FnPointer {
                    num_binders: 0,
                    sig: FnSig { abi: (), safety: chalk_ir::Safety::Safe, variadic: false },
                    substitution: FnSubst(
                        Substitution::from_iter(Interner, sig_tys.clone()).shifted_in(Interner),
                    ),
                })
                .intern(Interner);
                let closure_id = self.db.intern_closure((self.owner, tgt_expr)).into();
                let closure_ty =
                    TyKind::Closure(closure_id, Substitution::from1(Interner, sig_ty.clone()))
                        .intern(Interner);

                // Eagerly try to relate the closure type with the expected
                // type, otherwise we often won't have enough information to
                // infer the body.
                self.deduce_closure_type_from_expectations(
                    tgt_expr,
                    &closure_ty,
                    &sig_ty,
                    expected,
                );

                // Now go through the argument patterns
                for (arg_pat, arg_ty) in args.iter().zip(sig_tys) {
                    self.infer_pat(*arg_pat, &arg_ty, BindingMode::default());
                }

                let prev_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
                let prev_ret_ty = mem::replace(&mut self.return_ty, ret_ty.clone());

                self.infer_expr_coerce(*body, &Expectation::has_type(ret_ty));

                self.diverges = prev_diverges;
                self.return_ty = prev_ret_ty;

                closure_ty
            }
            Expr::Call { callee, args } => {
                let callee_ty = self.infer_expr(*callee, &Expectation::none());
                let canonicalized = self.canonicalize(callee_ty.clone());
                let mut derefs = Autoderef::new(
                    self.db,
                    self.resolver.krate(),
                    InEnvironment {
                        goal: canonicalized.value.clone(),
                        environment: self.table.trait_env.env.clone(),
                    },
                );
                let res = derefs.by_ref().find_map(|(callee_deref_ty, _)| {
                    let ty = &canonicalized.decanonicalize_ty(&mut self.table, callee_deref_ty);
                    self.callable_sig(ty, args.len())
                });
                let (param_tys, ret_ty): (Vec<Ty>, Ty) = match res {
                    Some(res) => {
                        self.write_expr_adj(*callee, self.auto_deref_adjust_steps(&derefs));
                        res
                    }
                    None => (Vec::new(), self.err_ty()),
                };
                self.register_obligations_for_call(&callee_ty);

                let expected_inputs = self.expected_inputs_for_expected_output(
                    expected,
                    ret_ty.clone(),
                    param_tys.clone(),
                );

                self.check_call_arguments(args, &expected_inputs, &param_tys);
                self.normalize_associated_types_in(ret_ty)
            }
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
                let input_ty = self.infer_expr(*expr, &Expectation::none());

                let expected = expected.adjust_for_branches(&mut self.table);

                let result_ty = if arms.is_empty() {
                    TyKind::Never.intern(Interner)
                } else {
                    match &expected {
                        Expectation::HasType(ty) => ty.clone(),
                        _ => self.table.new_type_var(),
                    }
                };
                let mut coerce = CoerceMany::new(result_ty);

                let matchee_diverges = self.diverges;
                let mut all_arms_diverge = Diverges::Always;

                for arm in arms.iter() {
                    self.diverges = Diverges::Maybe;
                    let _pat_ty = self.infer_pat(arm.pat, &input_ty, BindingMode::default());
                    match arm.guard {
                        Some(MatchGuard::If { expr: guard_expr }) => {
                            self.infer_expr(
                                guard_expr,
                                &Expectation::has_type(
                                    TyKind::Scalar(Scalar::Bool).intern(Interner),
                                ),
                            );
                        }
                        Some(MatchGuard::IfLet { expr, pat }) => {
                            let input_ty = self.infer_expr(expr, &Expectation::none());
                            let _pat_ty = self.infer_pat(pat, &input_ty, BindingMode::default());
                        }
                        _ => {}
                    }

                    let arm_ty = self.infer_expr_inner(arm.expr, &expected);
                    all_arms_diverge &= self.diverges;
                    coerce.coerce(self, Some(arm.expr), &arm_ty);
                }

                self.diverges = matchee_diverges | all_arms_diverge;

                coerce.complete()
            }
            Expr::Path(p) => {
                // FIXME this could be more efficient...
                let resolver = resolver_for_expr(self.db.upcast(), self.owner, tgt_expr);
                self.infer_path(&resolver, p, tgt_expr.into()).unwrap_or_else(|| self.err_ty())
            }
            Expr::Continue { .. } => TyKind::Never.intern(Interner),
            Expr::Break { expr, label } => {
                let mut coerce = match find_breakable(&mut self.breakables, label.as_ref()) {
                    Some(ctxt) => {
                        // avoiding the borrowck
                        mem::replace(
                            &mut ctxt.coerce,
                            CoerceMany::new(self.result.standard_types.unknown.clone()),
                        )
                    }
                    None => CoerceMany::new(self.result.standard_types.unknown.clone()),
                };

                let val_ty = if let Some(expr) = *expr {
                    self.infer_expr(expr, &Expectation::none())
                } else {
                    TyBuilder::unit()
                };

                // FIXME: create a synthetic `()` during lowering so we have something to refer to here?
                coerce.coerce(self, *expr, &val_ty);

                if let Some(ctxt) = find_breakable(&mut self.breakables, label.as_ref()) {
                    ctxt.coerce = coerce;
                    ctxt.may_break = true;
                } else {
                    self.push_diagnostic(InferenceDiagnostic::BreakOutsideOfLoop {
                        expr: tgt_expr,
                    });
                };

                TyKind::Never.intern(Interner)
            }
            Expr::Return { expr } => {
                if let Some(expr) = expr {
                    self.infer_expr_coerce(*expr, &Expectation::has_type(self.return_ty.clone()));
                } else {
                    let unit = TyBuilder::unit();
                    let _ = self.coerce(Some(tgt_expr), &unit, &self.return_ty.clone());
                }
                TyKind::Never.intern(Interner)
            }
            Expr::Yield { expr } => {
                // FIXME: track yield type for coercion
                if let Some(expr) = expr {
                    self.infer_expr(*expr, &Expectation::none());
                }
                TyKind::Never.intern(Interner)
            }
            Expr::RecordLit { path, fields, spread } => {
                let (ty, def_id) = self.resolve_variant(path.as_deref(), false);
                if let Some(variant) = def_id {
                    self.write_variant_resolution(tgt_expr.into(), variant);
                }

                if let Some(t) = expected.only_has_type(&mut self.table) {
                    self.unify(&ty, &t);
                }

                let substs = ty
                    .as_adt()
                    .map(|(_, s)| s.clone())
                    .unwrap_or_else(|| Substitution::empty(Interner));
                let field_types = def_id.map(|it| self.db.field_types(it)).unwrap_or_default();
                let variant_data = def_id.map(|it| it.variant_data(self.db.upcast()));
                for field in fields.iter() {
                    let field_def =
                        variant_data.as_ref().and_then(|it| match it.field(&field.name) {
                            Some(local_id) => Some(FieldId { parent: def_id.unwrap(), local_id }),
                            None => {
                                self.push_diagnostic(InferenceDiagnostic::NoSuchField {
                                    expr: field.expr,
                                });
                                None
                            }
                        });
                    let field_ty = field_def.map_or(self.err_ty(), |it| {
                        field_types[it.local_id].clone().substitute(Interner, &substs)
                    });
                    self.infer_expr_coerce(field.expr, &Expectation::has_type(field_ty));
                }
                if let Some(expr) = spread {
                    self.infer_expr(*expr, &Expectation::has_type(ty.clone()));
                }
                ty
            }
            Expr::Field { expr, name } => {
                let receiver_ty = self.infer_expr_inner(*expr, &Expectation::none());
                let canonicalized = self.canonicalize(receiver_ty);

                let mut autoderef = Autoderef::new(
                    self.db,
                    self.resolver.krate(),
                    InEnvironment {
                        goal: canonicalized.value.clone(),
                        environment: self.trait_env.env.clone(),
                    },
                );
                let ty = autoderef.by_ref().find_map(|(derefed_ty, _)| {
                    let module = self.resolver.module();
                    let db = self.db;
                    let is_visible = |field_id: &FieldId| {
                        module
                            .map(|mod_id| {
                                db.field_visibilities(field_id.parent)[field_id.local_id]
                                    .is_visible_from(db.upcast(), mod_id)
                            })
                            .unwrap_or(true)
                    };
                    match canonicalized
                        .decanonicalize_ty(&mut self.table, derefed_ty)
                        .kind(Interner)
                    {
                        TyKind::Tuple(_, substs) => name.as_tuple_index().and_then(|idx| {
                            substs
                                .as_slice(Interner)
                                .get(idx)
                                .map(|a| a.assert_ty_ref(Interner))
                                .cloned()
                        }),
                        TyKind::Adt(AdtId(hir_def::AdtId::StructId(s)), parameters) => {
                            let local_id = self.db.struct_data(*s).variant_data.field(name)?;
                            let field = FieldId { parent: (*s).into(), local_id };
                            if is_visible(&field) {
                                self.write_field_resolution(tgt_expr, field);
                                Some(
                                    self.db.field_types((*s).into())[field.local_id]
                                        .clone()
                                        .substitute(Interner, &parameters),
                                )
                            } else {
                                None
                            }
                        }
                        TyKind::Adt(AdtId(hir_def::AdtId::UnionId(u)), parameters) => {
                            let local_id = self.db.union_data(*u).variant_data.field(name)?;
                            let field = FieldId { parent: (*u).into(), local_id };
                            if is_visible(&field) {
                                self.write_field_resolution(tgt_expr, field);
                                Some(
                                    self.db.field_types((*u).into())[field.local_id]
                                        .clone()
                                        .substitute(Interner, &parameters),
                                )
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                });
                let ty = match ty {
                    Some(ty) => {
                        self.write_expr_adj(*expr, self.auto_deref_adjust_steps(&autoderef));
                        ty
                    }
                    None => self.err_ty(),
                };
                let ty = self.insert_type_vars(ty);
                self.normalize_associated_types_in(ty)
            }
            Expr::Await { expr } => {
                let inner_ty = self.infer_expr_inner(*expr, &Expectation::none());
                self.resolve_associated_type(inner_ty, self.resolve_future_future_output())
            }
            Expr::Try { expr } => {
                let inner_ty = self.infer_expr_inner(*expr, &Expectation::none());
                self.resolve_associated_type(inner_ty, self.resolve_ops_try_ok())
            }
            Expr::Cast { expr, type_ref } => {
                // FIXME: propagate the "castable to" expectation (and find a test case that shows this is necessary)
                let _inner_ty = self.infer_expr_inner(*expr, &Expectation::none());
                let cast_ty = self.make_ty(type_ref);
                // FIXME check the cast...
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
                    Expectation::rvalue_hint(&mut self.table, Ty::clone(exp_inner))
                } else {
                    Expectation::none()
                };
                let inner_ty = self.infer_expr_inner(*expr, &expectation);
                match rawness {
                    Rawness::RawPtr => TyKind::Raw(mutability, inner_ty),
                    Rawness::Ref => TyKind::Ref(mutability, static_lifetime(), inner_ty),
                }
                .intern(Interner)
            }
            Expr::Box { expr } => {
                let inner_ty = self.infer_expr_inner(*expr, &Expectation::none());
                if let Some(box_) = self.resolve_boxed_box() {
                    TyBuilder::adt(self.db, box_)
                        .push(inner_ty)
                        .fill_with_defaults(self.db, || self.table.new_type_var())
                        .build()
                } else {
                    self.err_ty()
                }
            }
            Expr::UnaryOp { expr, op } => {
                let inner_ty = self.infer_expr_inner(*expr, &Expectation::none());
                let inner_ty = self.resolve_ty_shallow(&inner_ty);
                match op {
                    UnaryOp::Deref => match self.resolver.krate() {
                        Some(krate) => {
                            let canonicalized = self.canonicalize(inner_ty);
                            match autoderef::deref(
                                self.db,
                                krate,
                                InEnvironment {
                                    goal: &canonicalized.value,
                                    environment: self.trait_env.env.clone(),
                                },
                            ) {
                                Some(derefed_ty) => {
                                    canonicalized.decanonicalize_ty(&mut self.table, derefed_ty)
                                }
                                None => self.err_ty(),
                            }
                        }
                        None => self.err_ty(),
                    },
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
                Some(BinaryOp::Assignment { op: None }) => {
                    let lhs_ty = self.infer_expr(*lhs, &Expectation::none());
                    self.infer_expr_coerce(*rhs, &Expectation::has_type(lhs_ty));
                    self.result.standard_types.unit.clone()
                }
                Some(BinaryOp::LogicOp(_)) => {
                    let bool_ty = self.result.standard_types.bool_.clone();
                    self.infer_expr_coerce(*lhs, &Expectation::HasType(bool_ty.clone()));
                    let lhs_diverges = self.diverges;
                    self.infer_expr_coerce(*rhs, &Expectation::HasType(bool_ty.clone()));
                    // Depending on the LHS' value, the RHS can never execute.
                    self.diverges = lhs_diverges;
                    bool_ty
                }
                Some(op) => self.infer_overloadable_binop(*lhs, *op, *rhs, tgt_expr),
                _ => self.err_ty(),
            },
            Expr::Range { lhs, rhs, range_type } => {
                let lhs_ty = lhs.map(|e| self.infer_expr_inner(e, &Expectation::none()));
                let rhs_expect = lhs_ty
                    .as_ref()
                    .map_or_else(Expectation::none, |ty| Expectation::has_type(ty.clone()));
                let rhs_ty = rhs.map(|e| self.infer_expr(e, &rhs_expect));
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
                let base_ty = self.infer_expr_inner(*base, &Expectation::none());
                let index_ty = self.infer_expr(*index, &Expectation::none());

                if let (Some(index_trait), Some(krate)) =
                    (self.resolve_ops_index(), self.resolver.krate())
                {
                    let canonicalized = self.canonicalize(base_ty);
                    let self_ty = method_resolution::resolve_indexing_op(
                        self.db,
                        &canonicalized.value,
                        self.trait_env.clone(),
                        krate,
                        index_trait,
                    );
                    let self_ty = self_ty.map_or(self.err_ty(), |t| {
                        canonicalized.decanonicalize_ty(&mut self.table, t)
                    });
                    self.resolve_associated_type_with_params(
                        self_ty,
                        self.resolve_ops_index_output(),
                        &[index_ty],
                    )
                } else {
                    self.err_ty()
                }
            }
            Expr::Tuple { exprs } => {
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
                    self.infer_expr_coerce(*expr, &Expectation::has_type(ty.clone()));
                }

                TyKind::Tuple(tys.len(), Substitution::from_iter(Interner, tys)).intern(Interner)
            }
            Expr::Array(array) => {
                let elem_ty =
                    match expected.to_option(&mut self.table).as_ref().map(|t| t.kind(Interner)) {
                        Some(TyKind::Array(st, _) | TyKind::Slice(st)) => st.clone(),
                        _ => self.table.new_type_var(),
                    };
                let mut coerce = CoerceMany::new(elem_ty.clone());

                let expected = Expectation::has_type(elem_ty.clone());
                let len = match array {
                    Array::ElementList(items) => {
                        for &expr in items.iter() {
                            let cur_elem_ty = self.infer_expr_inner(expr, &expected);
                            coerce.coerce(self, Some(expr), &cur_elem_ty);
                        }
                        Some(items.len() as u64)
                    }
                    &Array::Repeat { initializer, repeat } => {
                        self.infer_expr_coerce(initializer, &Expectation::has_type(elem_ty));
                        self.infer_expr(
                            repeat,
                            &Expectation::has_type(
                                TyKind::Scalar(Scalar::Uint(UintTy::Usize)).intern(Interner),
                            ),
                        );

                        consteval::eval_usize(
                            repeat,
                            consteval::ConstEvalCtx {
                                exprs: &body.exprs,
                                pats: &body.pats,
                                local_data: Default::default(),
                                infer: &mut |x| self.infer_expr(x, &expected),
                            },
                        )
                    }
                };

                TyKind::Array(coerce.complete(), consteval::usize_const(len)).intern(Interner)
            }
            Expr::Literal(lit) => match lit {
                Literal::Bool(..) => TyKind::Scalar(Scalar::Bool).intern(Interner),
                Literal::String(..) => {
                    TyKind::Ref(Mutability::Not, static_lifetime(), TyKind::Str.intern(Interner))
                        .intern(Interner)
                }
                Literal::ByteString(bs) => {
                    let byte_type = TyKind::Scalar(Scalar::Uint(UintTy::U8)).intern(Interner);

                    let len = consteval::usize_const(Some(bs.len() as u64));

                    let array_type = TyKind::Array(byte_type, len).intern(Interner);
                    TyKind::Ref(Mutability::Not, static_lifetime(), array_type).intern(Interner)
                }
                Literal::Char(..) => TyKind::Scalar(Scalar::Char).intern(Interner),
                Literal::Int(_v, ty) => match ty {
                    Some(int_ty) => {
                        TyKind::Scalar(Scalar::Int(primitive::int_ty_from_builtin(*int_ty)))
                            .intern(Interner)
                    }
                    None => self.table.new_integer_var(),
                },
                Literal::Uint(_v, ty) => match ty {
                    Some(int_ty) => {
                        TyKind::Scalar(Scalar::Uint(primitive::uint_ty_from_builtin(*int_ty)))
                            .intern(Interner)
                    }
                    None => self.table.new_integer_var(),
                },
                Literal::Float(_v, ty) => match ty {
                    Some(float_ty) => {
                        TyKind::Scalar(Scalar::Float(primitive::float_ty_from_builtin(*float_ty)))
                            .intern(Interner)
                    }
                    None => self.table.new_float_var(),
                },
            },
            Expr::MacroStmts { tail } => self.infer_expr_inner(*tail, expected),
        };
        // use a new type variable if we got unknown here
        let ty = self.insert_type_vars_shallow(ty);
        self.write_expr_ty(tgt_expr, ty.clone());
        ty
    }

    fn infer_overloadable_binop(
        &mut self,
        lhs: ExprId,
        op: BinaryOp,
        rhs: ExprId,
        tgt_expr: ExprId,
    ) -> Ty {
        let lhs_expectation = Expectation::none();
        let lhs_ty = self.infer_expr(lhs, &lhs_expectation);
        let rhs_ty = self.table.new_type_var();

        let func = self.resolve_binop_method(op);
        let func = match func {
            Some(func) => func,
            None => {
                let rhs_ty = self.builtin_binary_op_rhs_expectation(op, lhs_ty.clone());
                let rhs_ty = self.infer_expr_coerce(rhs, &Expectation::from_option(rhs_ty));
                return self
                    .builtin_binary_op_return_ty(op, lhs_ty, rhs_ty)
                    .unwrap_or_else(|| self.err_ty());
            }
        };

        let subst = TyBuilder::subst_for_def(self.db, func)
            .push(lhs_ty.clone())
            .push(rhs_ty.clone())
            .build();
        self.write_method_resolution(tgt_expr, func, subst.clone());

        let method_ty = self.db.value_ty(func.into()).substitute(Interner, &subst);
        self.register_obligations_for_call(&method_ty);

        self.infer_expr_coerce(rhs, &Expectation::has_type(rhs_ty.clone()));

        let ret_ty = match method_ty.callable_sig(self.db) {
            Some(sig) => sig.ret().clone(),
            None => self.err_ty(),
        };

        let ret_ty = self.normalize_associated_types_in(ret_ty);

        // FIXME: record autoref adjustments

        // use knowledge of built-in binary ops, which can sometimes help inference
        if let Some(builtin_rhs) = self.builtin_binary_op_rhs_expectation(op, lhs_ty.clone()) {
            self.unify(&builtin_rhs, &rhs_ty);
        }
        if let Some(builtin_ret) = self.builtin_binary_op_return_ty(op, lhs_ty, rhs_ty) {
            self.unify(&builtin_ret, &ret_ty);
        }

        ret_ty
    }

    fn infer_block(
        &mut self,
        expr: ExprId,
        statements: &[Statement],
        tail: Option<ExprId>,
        expected: &Expectation,
    ) -> Ty {
        for stmt in statements {
            match stmt {
                Statement::Let { pat, type_ref, initializer, else_branch } => {
                    let decl_ty = type_ref
                        .as_ref()
                        .map(|tr| self.make_ty(tr))
                        .unwrap_or_else(|| self.err_ty());

                    // Always use the declared type when specified
                    let mut ty = decl_ty.clone();

                    if let Some(expr) = initializer {
                        let actual_ty =
                            self.infer_expr_coerce(*expr, &Expectation::has_type(decl_ty.clone()));
                        if decl_ty.is_unknown() {
                            ty = actual_ty;
                        }
                    }

                    if let Some(expr) = else_branch {
                        self.infer_expr_coerce(
                            *expr,
                            &Expectation::has_type(Ty::new(Interner, TyKind::Never)),
                        );
                    }

                    self.infer_pat(*pat, &ty, BindingMode::default());
                }
                Statement::Expr { expr, .. } => {
                    self.infer_expr(*expr, &Expectation::none());
                }
            }
        }

        if let Some(expr) = tail {
            self.infer_expr_coerce(expr, expected)
        } else {
            // Citing rustc: if there is no explicit tail expression,
            // that is typically equivalent to a tail expression
            // of `()` -- except if the block diverges. In that
            // case, there is no value supplied from the tail
            // expression (assuming there are no other breaks,
            // this implies that the type of the block will be
            // `!`).
            if self.diverges.is_always() {
                // we don't even make an attempt at coercion
                self.table.new_maybe_never_var()
            } else {
                if let Some(t) = expected.only_has_type(&mut self.table) {
                    let _ = self.coerce(Some(expr), &TyBuilder::unit(), &t);
                }
                TyBuilder::unit()
            }
        }
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
        let receiver_ty = self.infer_expr(receiver, &Expectation::none());
        let canonicalized_receiver = self.canonicalize(receiver_ty.clone());

        let traits_in_scope = self.resolver.traits_in_scope(self.db.upcast());

        let resolved = self.resolver.krate().and_then(|krate| {
            method_resolution::lookup_method(
                &canonicalized_receiver.value,
                self.db,
                self.trait_env.clone(),
                krate,
                &traits_in_scope,
                self.resolver.module(),
                method_name,
            )
        });
        let (receiver_ty, method_ty, substs) = match resolved {
            Some((ty, func)) => {
                let ty = canonicalized_receiver.decanonicalize_ty(&mut self.table, ty);
                let generics = generics(self.db.upcast(), func.into());
                let substs = self.substs_for_method_call(generics, generic_args, &ty);
                self.write_method_resolution(tgt_expr, func, substs.clone());
                (ty, self.db.value_ty(func.into()), substs)
            }
            None => (
                receiver_ty,
                Binders::empty(Interner, self.err_ty()),
                Substitution::empty(Interner),
            ),
        };
        let method_ty = method_ty.substitute(Interner, &substs);
        self.register_obligations_for_call(&method_ty);
        let (formal_receiver_ty, param_tys, ret_ty) = match method_ty.callable_sig(self.db) {
            Some(sig) => {
                if !sig.params().is_empty() {
                    (sig.params()[0].clone(), sig.params()[1..].to_vec(), sig.ret().clone())
                } else {
                    (self.err_ty(), Vec::new(), sig.ret().clone())
                }
            }
            None => (self.err_ty(), Vec::new(), self.err_ty()),
        };
        self.unify(&formal_receiver_ty, &receiver_ty);

        let expected_inputs =
            self.expected_inputs_for_expected_output(expected, ret_ty.clone(), param_tys.clone());

        self.check_call_arguments(args, &expected_inputs, &param_tys);
        self.normalize_associated_types_in(ret_ty)
    }

    fn expected_inputs_for_expected_output(
        &mut self,
        expected_output: &Expectation,
        output: Ty,
        inputs: Vec<Ty>,
    ) -> Vec<Ty> {
        if let Some(expected_ty) = expected_output.to_option(&mut self.table) {
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

    fn check_call_arguments(&mut self, args: &[ExprId], expected_inputs: &[Ty], param_tys: &[Ty]) {
        // Quoting https://github.com/rust-lang/rust/blob/6ef275e6c3cb1384ec78128eceeb4963ff788dca/src/librustc_typeck/check/mod.rs#L3325 --
        // We do this in a pretty awful way: first we type-check any arguments
        // that are not closures, then we type-check the closures. This is so
        // that we have more information about the types of arguments when we
        // type-check the functions. This isn't really the right way to do this.
        for &check_closures in &[false, true] {
            let param_iter = param_tys.iter().cloned().chain(repeat(self.err_ty()));
            let expected_iter = expected_inputs
                .iter()
                .cloned()
                .chain(param_iter.clone().skip(expected_inputs.len()));
            for ((&arg, param_ty), expected_ty) in args.iter().zip(param_iter).zip(expected_iter) {
                let is_closure = matches!(&self.body[arg], Expr::Lambda { .. });
                if is_closure != check_closures {
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
                let expected = Expectation::rvalue_hint(&mut self.table, expected_ty);
                // infer with the expected type we have...
                let ty = self.infer_expr_inner(arg, &expected);

                // then coerce to either the expected type or just the formal parameter type
                let coercion_target = if let Some(ty) = expected.only_has_type(&mut self.table) {
                    // if we are coercing to the expectation, unify with the
                    // formal parameter type to connect everything
                    self.unify(&ty, &param_ty);
                    ty
                } else {
                    param_ty
                };
                if !coercion_target.is_unknown() {
                    if self.coerce(Some(arg), &ty, &coercion_target).is_err() {
                        self.result.type_mismatches.insert(
                            arg.into(),
                            TypeMismatch { expected: coercion_target, actual: ty.clone() },
                        );
                    }
                }
            }
        }
    }

    fn substs_for_method_call(
        &mut self,
        def_generics: Generics,
        generic_args: Option<&GenericArgs>,
        receiver_ty: &Ty,
    ) -> Substitution {
        let (parent_params, self_params, type_params, impl_trait_params) =
            def_generics.provenance_split();
        assert_eq!(self_params, 0); // method shouldn't have another Self param
        let total_len = parent_params + type_params + impl_trait_params;
        let mut substs = Vec::with_capacity(total_len);
        // Parent arguments are unknown, except for the receiver type
        for (_id, param) in def_generics.iter_parent() {
            if param.provenance == hir_def::generics::TypeParamProvenance::TraitSelf {
                substs.push(receiver_ty.clone());
            } else {
                substs.push(self.table.new_type_var());
            }
        }
        // handle provided type arguments
        if let Some(generic_args) = generic_args {
            // if args are provided, it should be all of them, but we can't rely on that
            for arg in generic_args
                .args
                .iter()
                .filter(|arg| matches!(arg, GenericArg::Type(_)))
                .take(type_params)
            {
                match arg {
                    GenericArg::Type(type_ref) => {
                        let ty = self.make_ty(type_ref);
                        substs.push(ty);
                    }
                    GenericArg::Lifetime(_) => {}
                }
            }
        };
        let supplied_params = substs.len();
        for _ in supplied_params..total_len {
            substs.push(self.table.new_type_var());
        }
        assert_eq!(substs.len(), total_len);
        Substitution::from_iter(Interner, substs)
    }

    fn register_obligations_for_call(&mut self, callable_ty: &Ty) {
        let callable_ty = self.resolve_ty_shallow(callable_ty);
        if let TyKind::FnDef(fn_def, parameters) = callable_ty.kind(Interner) {
            let def: CallableDefId = from_chalk(self.db, *fn_def);
            let generic_predicates = self.db.generic_predicates(def.into());
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
                    if let ItemContainerId::TraitId(trait_) = f.lookup(self.db.upcast()).container {
                        // construct a TraitRef
                        let substs = crate::subst_prefix(
                            &*parameters,
                            generics(self.db.upcast(), trait_.into()).len(),
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

    fn builtin_binary_op_return_ty(&mut self, op: BinaryOp, lhs_ty: Ty, rhs_ty: Ty) -> Option<Ty> {
        let lhs_ty = self.resolve_ty_shallow(&lhs_ty);
        let rhs_ty = self.resolve_ty_shallow(&rhs_ty);
        match op {
            BinaryOp::LogicOp(_) | BinaryOp::CmpOp(_) => {
                Some(TyKind::Scalar(Scalar::Bool).intern(Interner))
            }
            BinaryOp::Assignment { .. } => Some(TyBuilder::unit()),
            BinaryOp::ArithOp(ArithOp::Shl | ArithOp::Shr) => {
                // all integer combinations are valid here
                if matches!(
                    lhs_ty.kind(Interner),
                    TyKind::Scalar(Scalar::Int(_) | Scalar::Uint(_))
                        | TyKind::InferenceVar(_, TyVariableKind::Integer)
                ) && matches!(
                    rhs_ty.kind(Interner),
                    TyKind::Scalar(Scalar::Int(_) | Scalar::Uint(_))
                        | TyKind::InferenceVar(_, TyVariableKind::Integer)
                ) {
                    Some(lhs_ty)
                } else {
                    None
                }
            }
            BinaryOp::ArithOp(_) => match (lhs_ty.kind(Interner), rhs_ty.kind(Interner)) {
                // (int, int) | (uint, uint) | (float, float)
                (TyKind::Scalar(Scalar::Int(_)), TyKind::Scalar(Scalar::Int(_)))
                | (TyKind::Scalar(Scalar::Uint(_)), TyKind::Scalar(Scalar::Uint(_)))
                | (TyKind::Scalar(Scalar::Float(_)), TyKind::Scalar(Scalar::Float(_))) => {
                    Some(rhs_ty)
                }
                // ({int}, int) | ({int}, uint)
                (
                    TyKind::InferenceVar(_, TyVariableKind::Integer),
                    TyKind::Scalar(Scalar::Int(_) | Scalar::Uint(_)),
                ) => Some(rhs_ty),
                // (int, {int}) | (uint, {int})
                (
                    TyKind::Scalar(Scalar::Int(_) | Scalar::Uint(_)),
                    TyKind::InferenceVar(_, TyVariableKind::Integer),
                ) => Some(lhs_ty),
                // ({float} | float)
                (
                    TyKind::InferenceVar(_, TyVariableKind::Float),
                    TyKind::Scalar(Scalar::Float(_)),
                ) => Some(rhs_ty),
                // (float, {float})
                (
                    TyKind::Scalar(Scalar::Float(_)),
                    TyKind::InferenceVar(_, TyVariableKind::Float),
                ) => Some(lhs_ty),
                // ({int}, {int}) | ({float}, {float})
                (
                    TyKind::InferenceVar(_, TyVariableKind::Integer),
                    TyKind::InferenceVar(_, TyVariableKind::Integer),
                )
                | (
                    TyKind::InferenceVar(_, TyVariableKind::Float),
                    TyKind::InferenceVar(_, TyVariableKind::Float),
                ) => Some(rhs_ty),
                _ => None,
            },
        }
    }

    fn builtin_binary_op_rhs_expectation(&mut self, op: BinaryOp, lhs_ty: Ty) -> Option<Ty> {
        Some(match op {
            BinaryOp::LogicOp(..) => TyKind::Scalar(Scalar::Bool).intern(Interner),
            BinaryOp::Assignment { op: None } => lhs_ty,
            BinaryOp::CmpOp(CmpOp::Eq { .. }) => match self
                .resolve_ty_shallow(&lhs_ty)
                .kind(Interner)
            {
                TyKind::Scalar(_) | TyKind::Str => lhs_ty,
                TyKind::InferenceVar(_, TyVariableKind::Integer | TyVariableKind::Float) => lhs_ty,
                _ => return None,
            },
            BinaryOp::ArithOp(ArithOp::Shl | ArithOp::Shr) => return None,
            BinaryOp::CmpOp(CmpOp::Ord { .. })
            | BinaryOp::Assignment { op: Some(_) }
            | BinaryOp::ArithOp(_) => match self.resolve_ty_shallow(&lhs_ty).kind(Interner) {
                TyKind::Scalar(Scalar::Int(_) | Scalar::Uint(_) | Scalar::Float(_)) => lhs_ty,
                TyKind::InferenceVar(_, TyVariableKind::Integer | TyVariableKind::Float) => lhs_ty,
                _ => return None,
            },
        })
    }

    fn resolve_binop_method(&self, op: BinaryOp) -> Option<FunctionId> {
        let (name, lang_item) = match op {
            BinaryOp::LogicOp(_) => return None,
            BinaryOp::ArithOp(aop) => match aop {
                ArithOp::Add => (name!(add), name!(add)),
                ArithOp::Mul => (name!(mul), name!(mul)),
                ArithOp::Sub => (name!(sub), name!(sub)),
                ArithOp::Div => (name!(div), name!(div)),
                ArithOp::Rem => (name!(rem), name!(rem)),
                ArithOp::Shl => (name!(shl), name!(shl)),
                ArithOp::Shr => (name!(shr), name!(shr)),
                ArithOp::BitXor => (name!(bitxor), name!(bitxor)),
                ArithOp::BitOr => (name!(bitor), name!(bitor)),
                ArithOp::BitAnd => (name!(bitand), name!(bitand)),
            },
            BinaryOp::Assignment { op: Some(aop) } => match aop {
                ArithOp::Add => (name!(add_assign), name!(add_assign)),
                ArithOp::Mul => (name!(mul_assign), name!(mul_assign)),
                ArithOp::Sub => (name!(sub_assign), name!(sub_assign)),
                ArithOp::Div => (name!(div_assign), name!(div_assign)),
                ArithOp::Rem => (name!(rem_assign), name!(rem_assign)),
                ArithOp::Shl => (name!(shl_assign), name!(shl_assign)),
                ArithOp::Shr => (name!(shr_assign), name!(shr_assign)),
                ArithOp::BitXor => (name!(bitxor_assign), name!(bitxor_assign)),
                ArithOp::BitOr => (name!(bitor_assign), name!(bitor_assign)),
                ArithOp::BitAnd => (name!(bitand_assign), name!(bitand_assign)),
            },
            BinaryOp::CmpOp(cop) => match cop {
                CmpOp::Eq { negated: false } => (name!(eq), name!(eq)),
                CmpOp::Eq { negated: true } => (name!(ne), name!(eq)),
                CmpOp::Ord { ordering: Ordering::Less, strict: false } => {
                    (name!(le), name!(partial_ord))
                }
                CmpOp::Ord { ordering: Ordering::Less, strict: true } => {
                    (name!(lt), name!(partial_ord))
                }
                CmpOp::Ord { ordering: Ordering::Greater, strict: false } => {
                    (name!(ge), name!(partial_ord))
                }
                CmpOp::Ord { ordering: Ordering::Greater, strict: true } => {
                    (name!(gt), name!(partial_ord))
                }
            },
            BinaryOp::Assignment { op: None } => return None,
        };

        let trait_ = self.resolve_lang_item(lang_item)?.as_trait()?;

        self.db.trait_data(trait_).method_by_name(&name)
    }
}
