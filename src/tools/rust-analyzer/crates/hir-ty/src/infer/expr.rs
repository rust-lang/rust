//! Type inference for expressions.

use std::{
    iter::{repeat, repeat_with},
    mem,
};

use chalk_ir::{
    cast::Cast, fold::Shift, DebruijnIndex, GenericArgData, Mutability, TyKind, TyVariableKind,
};
use hir_def::{
    expr::{
        ArithOp, Array, BinaryOp, ClosureKind, Expr, ExprId, LabelId, Literal, Statement, UnaryOp,
    },
    generics::TypeOrConstParamData,
    lang_item::LangItem,
    path::{GenericArg, GenericArgs},
    resolver::resolver_for_expr,
    ConstParamId, FieldId, ItemContainerId, Lookup,
};
use hir_expand::name::{name, Name};
use stdx::always;
use syntax::ast::RangeOp;

use crate::{
    autoderef::{self, Autoderef},
    consteval,
    infer::{coerce::CoerceMany, find_continuable, BreakableKind},
    lower::{
        const_or_path_to_chalk, generic_arg_to_chalk, lower_to_chalk_mutability, ParamLoweringMode,
    },
    mapping::{from_chalk, ToChalk},
    method_resolution::{self, lang_items_for_bin_op, VisibleFromModule},
    primitive::{self, UintTy},
    static_lifetime, to_chalk_trait_id,
    utils::{generics, Generics},
    Adjust, Adjustment, AdtId, AutoBorrow, Binders, CallableDefId, FnPointer, FnSig, FnSubst,
    Interner, Rawness, Scalar, Substitution, TraitRef, Ty, TyBuilder, TyExt,
};

use super::{
    coerce::auto_deref_adjust_steps, find_breakable, BindingMode, BreakableContext, Diverges,
    Expectation, InferenceContext, InferenceDiagnostic, TypeMismatch,
};

impl<'a> InferenceContext<'a> {
    pub(crate) fn infer_expr(&mut self, tgt_expr: ExprId, expected: &Expectation) -> Ty {
        let ty = self.infer_expr_inner(tgt_expr, expected);
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

    fn infer_expr_inner(&mut self, tgt_expr: ExprId, expected: &Expectation) -> Ty {
        self.db.unwind_if_cancelled();

        let ty = match &self.body[tgt_expr] {
            Expr::Missing => self.err_ty(),
            &Expr::If { condition, then_branch, else_branch } => {
                let expected = &expected.adjust_for_branches(&mut self.table);
                self.infer_expr(
                    condition,
                    &Expectation::HasType(self.result.standard_types.bool_.clone()),
                );

                let condition_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
                let mut both_arms_diverge = Diverges::Always;

                let then_ty = self.infer_expr_inner(then_branch, expected);
                both_arms_diverge &= mem::replace(&mut self.diverges, Diverges::Maybe);
                let mut coerce = CoerceMany::new(expected.coercion_target_type(&mut self.table));
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
            &Expr::Let { pat, expr } => {
                let input_ty = self.infer_expr(expr, &Expectation::none());
                self.infer_pat(pat, &input_ty, BindingMode::default());
                self.result.standard_types.bool_.clone()
            }
            Expr::Block { statements, tail, label, id: _ } => {
                let old_resolver = mem::replace(
                    &mut self.resolver,
                    resolver_for_expr(self.db.upcast(), self.owner, tgt_expr),
                );
                let ty = match label {
                    Some(_) => {
                        let break_ty = self.table.new_type_var();
                        let (breaks, ty) = self.with_breakable_ctx(
                            BreakableKind::Block,
                            break_ty.clone(),
                            *label,
                            |this| {
                                this.infer_block(
                                    tgt_expr,
                                    statements,
                                    *tail,
                                    &Expectation::has_type(break_ty),
                                )
                            },
                        );
                        breaks.unwrap_or(ty)
                    }
                    None => self.infer_block(tgt_expr, statements, *tail, expected),
                };
                self.resolver = old_resolver;
                ty
            }
            Expr::Unsafe { body } => self.infer_expr(*body, expected),
            Expr::Const { body } => {
                self.with_breakable_ctx(BreakableKind::Border, self.err_ty(), None, |this| {
                    this.infer_expr(*body, expected)
                })
                .1
            }
            Expr::TryBlock { body } => {
                // The type that is returned from the try block
                let try_ty = self.table.new_type_var();
                if let Some(ty) = expected.only_has_type(&mut self.table) {
                    self.unify(&try_ty, &ty);
                }

                // The ok-ish type that is expected from the last expression
                let ok_ty =
                    self.resolve_associated_type(try_ty.clone(), self.resolve_ops_try_output());

                self.with_breakable_ctx(BreakableKind::Block, ok_ty.clone(), None, |this| {
                    this.infer_expr(*body, &Expectation::has_type(ok_ty));
                });

                try_ty
            }
            Expr::Async { body } => {
                let ret_ty = self.table.new_type_var();
                let prev_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
                let prev_ret_ty = mem::replace(&mut self.return_ty, ret_ty.clone());

                let (_, inner_ty) =
                    self.with_breakable_ctx(BreakableKind::Border, self.err_ty(), None, |this| {
                        this.infer_expr_coerce(*body, &Expectation::has_type(ret_ty))
                    });

                self.diverges = prev_diverges;
                self.return_ty = prev_ret_ty;

                // Use the first type parameter as the output type of future.
                // existential type AsyncBlockImplTrait<InnerType>: Future<Output = InnerType>
                let impl_trait_id = crate::ImplTraitId::AsyncBlockTypeImplTrait(self.owner, *body);
                let opaque_ty_id = self.db.intern_impl_trait_id(impl_trait_id).into();
                TyKind::OpaqueType(opaque_ty_id, Substitution::from1(Interner, inner_ty))
                    .intern(Interner)
            }
            &Expr::Loop { body, label } => {
                // FIXME: should be:
                // let ty = expected.coercion_target_type(&mut self.table);
                let ty = self.table.new_type_var();
                let (breaks, ()) =
                    self.with_breakable_ctx(BreakableKind::Loop, ty, label, |this| {
                        this.infer_expr(body, &Expectation::HasType(TyBuilder::unit()));
                    });

                match breaks {
                    Some(breaks) => {
                        self.diverges = Diverges::Maybe;
                        breaks
                    }
                    None => self.result.standard_types.never.clone(),
                }
            }
            &Expr::While { condition, body, label } => {
                self.with_breakable_ctx(BreakableKind::Loop, self.err_ty(), label, |this| {
                    this.infer_expr(
                        condition,
                        &Expectation::HasType(this.result.standard_types.bool_.clone()),
                    );
                    this.infer_expr(body, &Expectation::HasType(TyBuilder::unit()));
                });

                // the body may not run, so it diverging doesn't mean we diverge
                self.diverges = Diverges::Maybe;
                TyBuilder::unit()
            }
            &Expr::For { iterable, body, pat, label } => {
                let iterable_ty = self.infer_expr(iterable, &Expectation::none());
                let into_iter_ty =
                    self.resolve_associated_type(iterable_ty, self.resolve_into_iter_item());
                let pat_ty =
                    self.resolve_associated_type(into_iter_ty, self.resolve_iterator_item());

                self.infer_pat(pat, &pat_ty, BindingMode::default());
                self.with_breakable_ctx(BreakableKind::Loop, self.err_ty(), label, |this| {
                    this.infer_expr(body, &Expectation::HasType(TyBuilder::unit()));
                });

                // the body may not run, so it diverging doesn't mean we diverge
                self.diverges = Diverges::Maybe;
                TyBuilder::unit()
            }
            Expr::Closure { body, args, ret_type, arg_types, closure_kind } => {
                assert_eq!(args.len(), arg_types.len());

                let mut sig_tys = Vec::with_capacity(arg_types.len() + 1);

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
                        Substitution::from_iter(Interner, sig_tys.iter().cloned())
                            .shifted_in(Interner),
                    ),
                })
                .intern(Interner);

                let (ty, resume_yield_tys) = if matches!(closure_kind, ClosureKind::Generator(_)) {
                    // FIXME: report error when there are more than 1 parameter.
                    let resume_ty = match sig_tys.first() {
                        // When `sig_tys.len() == 1` the first type is the return type, not the
                        // first parameter type.
                        Some(ty) if sig_tys.len() > 1 => ty.clone(),
                        _ => self.result.standard_types.unit.clone(),
                    };
                    let yield_ty = self.table.new_type_var();

                    let subst = TyBuilder::subst_for_generator(self.db, self.owner)
                        .push(resume_ty.clone())
                        .push(yield_ty.clone())
                        .push(ret_ty.clone())
                        .build();

                    let generator_id = self.db.intern_generator((self.owner, tgt_expr)).into();
                    let generator_ty = TyKind::Generator(generator_id, subst).intern(Interner);

                    (generator_ty, Some((resume_ty, yield_ty)))
                } else {
                    let closure_id = self.db.intern_closure((self.owner, tgt_expr)).into();
                    let closure_ty =
                        TyKind::Closure(closure_id, Substitution::from1(Interner, sig_ty.clone()))
                            .intern(Interner);

                    (closure_ty, None)
                };

                // Eagerly try to relate the closure type with the expected
                // type, otherwise we often won't have enough information to
                // infer the body.
                self.deduce_closure_type_from_expectations(tgt_expr, &ty, &sig_ty, expected);

                // Now go through the argument patterns
                for (arg_pat, arg_ty) in args.iter().zip(sig_tys) {
                    self.infer_pat(*arg_pat, &arg_ty, BindingMode::default());
                }

                let prev_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
                let prev_ret_ty = mem::replace(&mut self.return_ty, ret_ty.clone());
                let prev_resume_yield_tys =
                    mem::replace(&mut self.resume_yield_tys, resume_yield_tys);

                self.with_breakable_ctx(BreakableKind::Border, self.err_ty(), None, |this| {
                    this.infer_expr_coerce(*body, &Expectation::has_type(ret_ty));
                });

                self.diverges = prev_diverges;
                self.return_ty = prev_ret_ty;
                self.resume_yield_tys = prev_resume_yield_tys;

                ty
            }
            Expr::Call { callee, args, .. } => {
                let callee_ty = self.infer_expr(*callee, &Expectation::none());
                let mut derefs = Autoderef::new(&mut self.table, callee_ty.clone());
                let (res, derefed_callee) = 'b: {
                    // manual loop to be able to access `derefs.table`
                    while let Some((callee_deref_ty, _)) = derefs.next() {
                        let res = derefs.table.callable_sig(&callee_deref_ty, args.len());
                        if res.is_some() {
                            break 'b (res, callee_deref_ty);
                        }
                    }
                    (None, callee_ty.clone())
                };
                // if the function is unresolved, we use is_varargs=true to
                // suppress the arg count diagnostic here
                let is_varargs =
                    derefed_callee.callable_sig(self.db).map_or(false, |sig| sig.is_varargs)
                        || res.is_none();
                let (param_tys, ret_ty) = match res {
                    Some((func, params, ret_ty)) => {
                        let adjustments = auto_deref_adjust_steps(&derefs);
                        // FIXME: Handle call adjustments for Fn/FnMut
                        self.write_expr_adj(*callee, adjustments);
                        if let Some((trait_, func)) = func {
                            let subst = TyBuilder::subst_for_def(self.db, trait_, None)
                                .push(callee_ty.clone())
                                .push(TyBuilder::tuple_with(params.iter().cloned()))
                                .build();
                            self.write_method_resolution(tgt_expr, func, subst.clone());
                        }
                        (params, ret_ty)
                    }
                    None => (Vec::new(), self.err_ty()), // FIXME diagnostic
                };
                let indices_to_skip = self.check_legacy_const_generics(derefed_callee, args);
                self.register_obligations_for_call(&callee_ty);

                let expected_inputs = self.expected_inputs_for_expected_output(
                    expected,
                    ret_ty.clone(),
                    param_tys.clone(),
                );

                self.check_call_arguments(
                    tgt_expr,
                    args,
                    &expected_inputs,
                    &param_tys,
                    &indices_to_skip,
                    is_varargs,
                );
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
                    self.result.standard_types.never.clone()
                } else {
                    expected.coercion_target_type(&mut self.table)
                };
                let mut coerce = CoerceMany::new(result_ty);

                let matchee_diverges = self.diverges;
                let mut all_arms_diverge = Diverges::Always;

                for arm in arms.iter() {
                    self.diverges = Diverges::Maybe;
                    let _pat_ty = self.infer_pat(arm.pat, &input_ty, BindingMode::default());
                    if let Some(guard_expr) = arm.guard {
                        self.infer_expr(
                            guard_expr,
                            &Expectation::HasType(self.result.standard_types.bool_.clone()),
                        );
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
            Expr::Continue { label } => {
                if let None = find_continuable(&mut self.breakables, label.as_ref()) {
                    self.push_diagnostic(InferenceDiagnostic::BreakOutsideOfLoop {
                        expr: tgt_expr,
                        is_break: false,
                    });
                };
                self.result.standard_types.never.clone()
            }
            Expr::Break { expr, label } => {
                let val_ty = if let Some(expr) = *expr {
                    self.infer_expr(expr, &Expectation::none())
                } else {
                    TyBuilder::unit()
                };

                match find_breakable(&mut self.breakables, label.as_ref()) {
                    Some(ctxt) => {
                        // avoiding the borrowck
                        let mut coerce = mem::replace(
                            &mut ctxt.coerce,
                            CoerceMany::new(expected.coercion_target_type(&mut self.table)),
                        );

                        // FIXME: create a synthetic `()` during lowering so we have something to refer to here?
                        coerce.coerce(self, *expr, &val_ty);

                        let ctxt = find_breakable(&mut self.breakables, label.as_ref())
                            .expect("breakable stack changed during coercion");
                        ctxt.coerce = coerce;
                        ctxt.may_break = true;
                    }
                    None => {
                        self.push_diagnostic(InferenceDiagnostic::BreakOutsideOfLoop {
                            expr: tgt_expr,
                            is_break: true,
                        });
                    }
                }
                self.result.standard_types.never.clone()
            }
            Expr::Return { expr } => {
                if let Some(expr) = expr {
                    self.infer_expr_coerce(*expr, &Expectation::has_type(self.return_ty.clone()));
                } else {
                    let unit = TyBuilder::unit();
                    let _ = self.coerce(Some(tgt_expr), &unit, &self.return_ty.clone());
                }
                self.result.standard_types.never.clone()
            }
            Expr::Yield { expr } => {
                if let Some((resume_ty, yield_ty)) = self.resume_yield_tys.clone() {
                    if let Some(expr) = expr {
                        self.infer_expr_coerce(*expr, &Expectation::has_type(yield_ty));
                    } else {
                        let unit = self.result.standard_types.unit.clone();
                        let _ = self.coerce(Some(tgt_expr), &unit, &yield_ty);
                    }
                    resume_ty
                } else {
                    // FIXME: report error (yield expr in non-generator)
                    self.result.standard_types.unknown.clone()
                }
            }
            Expr::Yeet { expr } => {
                if let &Some(expr) = expr {
                    self.infer_expr_inner(expr, &Expectation::None);
                }
                self.result.standard_types.never.clone()
            }
            Expr::RecordLit { path, fields, spread, .. } => {
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

                let mut autoderef = Autoderef::new(&mut self.table, receiver_ty);
                let mut private_field = None;
                let ty = autoderef.by_ref().find_map(|(derefed_ty, _)| {
                    let (field_id, parameters) = match derefed_ty.kind(Interner) {
                        TyKind::Tuple(_, substs) => {
                            return name.as_tuple_index().and_then(|idx| {
                                substs
                                    .as_slice(Interner)
                                    .get(idx)
                                    .map(|a| a.assert_ty_ref(Interner))
                                    .cloned()
                            });
                        }
                        TyKind::Adt(AdtId(hir_def::AdtId::StructId(s)), parameters) => {
                            let local_id = self.db.struct_data(*s).variant_data.field(name)?;
                            let field = FieldId { parent: (*s).into(), local_id };
                            (field, parameters.clone())
                        }
                        TyKind::Adt(AdtId(hir_def::AdtId::UnionId(u)), parameters) => {
                            let local_id = self.db.union_data(*u).variant_data.field(name)?;
                            let field = FieldId { parent: (*u).into(), local_id };
                            (field, parameters.clone())
                        }
                        _ => return None,
                    };
                    let is_visible = self.db.field_visibilities(field_id.parent)[field_id.local_id]
                        .is_visible_from(self.db.upcast(), self.resolver.module());
                    if !is_visible {
                        if private_field.is_none() {
                            private_field = Some(field_id);
                        }
                        return None;
                    }
                    // can't have `write_field_resolution` here because `self.table` is borrowed :(
                    self.result.field_resolutions.insert(tgt_expr, field_id);
                    let ty = self.db.field_types(field_id.parent)[field_id.local_id]
                        .clone()
                        .substitute(Interner, &parameters);
                    Some(ty)
                });
                let ty = match ty {
                    Some(ty) => {
                        let adjustments = auto_deref_adjust_steps(&autoderef);
                        self.write_expr_adj(*expr, adjustments);
                        let ty = self.insert_type_vars(ty);
                        let ty = self.normalize_associated_types_in(ty);
                        ty
                    }
                    _ => {
                        // Write down the first private field resolution if we found no field
                        // This aids IDE features for private fields like goto def
                        if let Some(field) = private_field {
                            self.result.field_resolutions.insert(tgt_expr, field);
                            self.result
                                .diagnostics
                                .push(InferenceDiagnostic::PrivateField { expr: tgt_expr, field });
                        }
                        self.err_ty()
                    }
                };
                ty
            }
            Expr::Await { expr } => {
                let inner_ty = self.infer_expr_inner(*expr, &Expectation::none());
                self.resolve_associated_type(inner_ty, self.resolve_future_future_output())
            }
            Expr::Try { expr } => {
                let inner_ty = self.infer_expr_inner(*expr, &Expectation::none());
                if let Some(trait_) = self.resolve_lang_trait(LangItem::Try) {
                    if let Some(func) = self.db.trait_data(trait_).method_by_name(&name!(branch)) {
                        let subst = TyBuilder::subst_for_def(self.db, trait_, None)
                            .push(inner_ty.clone())
                            .build();
                        self.write_method_resolution(tgt_expr, func, subst.clone());
                    }
                    let try_output = self.resolve_output_on(trait_);
                    self.resolve_associated_type(inner_ty, try_output)
                } else {
                    self.err_ty()
                }
            }
            Expr::Cast { expr, type_ref } => {
                let cast_ty = self.make_ty(type_ref);
                // FIXME: propagate the "castable to" expectation
                let _inner_ty = self.infer_expr_inner(*expr, &Expectation::None);
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
            &Expr::Box { expr } => self.infer_expr_box(expr, expected),
            Expr::UnaryOp { expr, op } => {
                let inner_ty = self.infer_expr_inner(*expr, &Expectation::none());
                let inner_ty = self.resolve_ty_shallow(&inner_ty);
                // FIXME: Note down method resolution her
                match op {
                    UnaryOp::Deref => {
                        autoderef::deref(&mut self.table, inner_ty).unwrap_or_else(|| self.err_ty())
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
                Some(BinaryOp::Assignment { op: None }) => {
                    let lhs = *lhs;
                    let is_ordinary = match &self.body[lhs] {
                        Expr::Array(_)
                        | Expr::RecordLit { .. }
                        | Expr::Tuple { .. }
                        | Expr::Underscore => false,
                        Expr::Call { callee, .. } => !matches!(&self.body[*callee], Expr::Path(_)),
                        _ => true,
                    };

                    // In ordinary (non-destructuring) assignments, the type of
                    // `lhs` must be inferred first so that the ADT fields
                    // instantiations in RHS can be coerced to it. Note that this
                    // cannot happen in destructuring assignments because of how
                    // they are desugared.
                    if is_ordinary {
                        let lhs_ty = self.infer_expr(lhs, &Expectation::none());
                        self.infer_expr_coerce(*rhs, &Expectation::has_type(lhs_ty));
                    } else {
                        let rhs_ty = self.infer_expr(*rhs, &Expectation::none());
                        self.infer_assignee_expr(lhs, &rhs_ty);
                    }
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

                if let Some(index_trait) = self.resolve_lang_trait(LangItem::Index) {
                    let canonicalized = self.canonicalize(base_ty.clone());
                    let receiver_adjustments = method_resolution::resolve_indexing_op(
                        self.db,
                        self.trait_env.clone(),
                        canonicalized.value,
                        index_trait,
                    );
                    let (self_ty, adj) = receiver_adjustments
                        .map_or((self.err_ty(), Vec::new()), |adj| {
                            adj.apply(&mut self.table, base_ty)
                        });
                    self.write_expr_adj(*base, adj);
                    if let Some(func) =
                        self.db.trait_data(index_trait).method_by_name(&name!(index))
                    {
                        let substs = TyBuilder::subst_for_def(self.db, index_trait, None)
                            .push(self_ty.clone())
                            .push(index_ty.clone())
                            .build();
                        self.write_method_resolution(tgt_expr, func, substs.clone());
                    }
                    self.resolve_associated_type_with_params(
                        self_ty,
                        self.resolve_ops_index_output(),
                        &[GenericArgData::Ty(index_ty).intern(Interner)],
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
                    Array::ElementList { elements, .. } => {
                        for &expr in elements.iter() {
                            let cur_elem_ty = self.infer_expr_inner(expr, &expected);
                            coerce.coerce(self, Some(expr), &cur_elem_ty);
                        }
                        consteval::usize_const(Some(elements.len() as u128))
                    }
                    &Array::Repeat { initializer, repeat } => {
                        self.infer_expr_coerce(initializer, &Expectation::has_type(elem_ty));
                        self.infer_expr(
                            repeat,
                            &Expectation::HasType(
                                TyKind::Scalar(Scalar::Uint(UintTy::Usize)).intern(Interner),
                            ),
                        );

                        if let Some(g_def) = self.owner.as_generic_def_id() {
                            let generics = generics(self.db.upcast(), g_def);
                            consteval::eval_to_const(
                                repeat,
                                ParamLoweringMode::Placeholder,
                                self,
                                || generics,
                                DebruijnIndex::INNERMOST,
                            )
                        } else {
                            consteval::usize_const(None)
                        }
                    }
                };

                TyKind::Array(coerce.complete(), len).intern(Interner)
            }
            Expr::Literal(lit) => match lit {
                Literal::Bool(..) => self.result.standard_types.bool_.clone(),
                Literal::String(..) => {
                    TyKind::Ref(Mutability::Not, static_lifetime(), TyKind::Str.intern(Interner))
                        .intern(Interner)
                }
                Literal::ByteString(bs) => {
                    let byte_type = TyKind::Scalar(Scalar::Uint(UintTy::U8)).intern(Interner);

                    let len = consteval::usize_const(Some(bs.len() as u128));

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
            Expr::Underscore => {
                // Underscore expressions may only appear in assignee expressions,
                // which are handled by `infer_assignee_expr()`, so any underscore
                // expression reaching this branch is an error.
                self.err_ty()
            }
        };
        // use a new type variable if we got unknown here
        let ty = self.insert_type_vars_shallow(ty);
        self.write_expr_ty(tgt_expr, ty.clone());
        if self.resolve_ty_shallow(&ty).is_never() {
            // Any expression that produces a value of type `!` must have diverged
            self.diverges = Diverges::Always;
        }
        ty
    }

    fn infer_expr_box(&mut self, inner_expr: ExprId, expected: &Expectation) -> Ty {
        if let Some(box_id) = self.resolve_boxed_box() {
            let table = &mut self.table;
            let inner_exp = expected
                .to_option(table)
                .as_ref()
                .map(|e| e.as_adt())
                .flatten()
                .filter(|(e_adt, _)| e_adt == &box_id)
                .map(|(_, subts)| {
                    let g = subts.at(Interner, 0);
                    Expectation::rvalue_hint(table, Ty::clone(g.assert_ty_ref(Interner)))
                })
                .unwrap_or_else(Expectation::none);

            let inner_ty = self.infer_expr_inner(inner_expr, &inner_exp);
            TyBuilder::adt(self.db, box_id)
                .push(inner_ty)
                .fill_with_defaults(self.db, || self.table.new_type_var())
                .build()
        } else {
            self.err_ty()
        }
    }

    pub(super) fn infer_assignee_expr(&mut self, lhs: ExprId, rhs_ty: &Ty) -> Ty {
        let is_rest_expr = |expr| {
            matches!(
                &self.body[expr],
                Expr::Range { lhs: None, rhs: None, range_type: RangeOp::Exclusive },
            )
        };

        let rhs_ty = self.resolve_ty_shallow(rhs_ty);

        let ty = match &self.body[lhs] {
            Expr::Tuple { exprs, .. } => {
                // We don't consider multiple ellipses. This is analogous to
                // `hir_def::body::lower::ExprCollector::collect_tuple_pat()`.
                let ellipsis = exprs.iter().position(|e| is_rest_expr(*e));
                let exprs: Vec<_> = exprs.iter().filter(|e| !is_rest_expr(**e)).copied().collect();

                self.infer_tuple_pat_like(&rhs_ty, (), ellipsis, &exprs)
            }
            Expr::Call { callee, args, .. } => {
                // Tuple structs
                let path = match &self.body[*callee] {
                    Expr::Path(path) => Some(path),
                    _ => None,
                };

                // We don't consider multiple ellipses. This is analogous to
                // `hir_def::body::lower::ExprCollector::collect_tuple_pat()`.
                let ellipsis = args.iter().position(|e| is_rest_expr(*e));
                let args: Vec<_> = args.iter().filter(|e| !is_rest_expr(**e)).copied().collect();

                self.infer_tuple_struct_pat_like(path, &rhs_ty, (), lhs, ellipsis, &args)
            }
            Expr::Array(Array::ElementList { elements, .. }) => {
                let elem_ty = match rhs_ty.kind(Interner) {
                    TyKind::Array(st, _) => st.clone(),
                    _ => self.err_ty(),
                };

                // There's no need to handle `..` as it cannot be bound.
                let sub_exprs = elements.iter().filter(|e| !is_rest_expr(**e));

                for e in sub_exprs {
                    self.infer_assignee_expr(*e, &elem_ty);
                }

                match rhs_ty.kind(Interner) {
                    TyKind::Array(_, _) => rhs_ty.clone(),
                    // Even when `rhs_ty` is not an array type, this assignee
                    // expression is inferred to be an array (of unknown element
                    // type and length). This should not be just an error type,
                    // because we are to compute the unifiability of this type and
                    // `rhs_ty` in the end of this function to issue type mismatches.
                    _ => TyKind::Array(self.err_ty(), crate::consteval::usize_const(None))
                        .intern(Interner),
                }
            }
            Expr::RecordLit { path, fields, .. } => {
                let subs = fields.iter().map(|f| (f.name.clone(), f.expr));

                self.infer_record_pat_like(path.as_deref(), &rhs_ty, (), lhs, subs)
            }
            Expr::Underscore => rhs_ty.clone(),
            _ => {
                // `lhs` is a place expression, a unit struct, or an enum variant.
                let lhs_ty = self.infer_expr(lhs, &Expectation::none());

                // This is the only branch where this function may coerce any type.
                // We are returning early to avoid the unifiability check below.
                let lhs_ty = self.insert_type_vars_shallow(lhs_ty);
                let ty = match self.coerce(None, &rhs_ty, &lhs_ty) {
                    Ok(ty) => ty,
                    Err(_) => {
                        self.result.type_mismatches.insert(
                            lhs.into(),
                            TypeMismatch { expected: rhs_ty.clone(), actual: lhs_ty.clone() },
                        );
                        // `rhs_ty` is returned so no further type mismatches are
                        // reported because of this mismatch.
                        rhs_ty
                    }
                };
                self.write_expr_ty(lhs, ty.clone());
                return ty;
            }
        };

        let ty = self.insert_type_vars_shallow(ty);
        if !self.unify(&ty, &rhs_ty) {
            self.result
                .type_mismatches
                .insert(lhs.into(), TypeMismatch { expected: rhs_ty.clone(), actual: ty.clone() });
        }
        self.write_expr_ty(lhs, ty.clone());
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

        let trait_func = lang_items_for_bin_op(op).and_then(|(name, lang_item)| {
            let trait_id = self.resolve_lang_item(lang_item)?.as_trait()?;
            let func = self.db.trait_data(trait_id).method_by_name(&name)?;
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

                self.infer_expr_coerce(rhs, &Expectation::has_type(rhs_ty));

                return ret_ty;
            }
        };

        // HACK: We can use this substitution for the function because the function itself doesn't
        // have its own generic parameters.
        let subst = TyBuilder::subst_for_def(self.db, trait_, None)
            .push(lhs_ty.clone())
            .push(rhs_ty.clone())
            .build();
        self.write_method_resolution(tgt_expr, func, subst.clone());

        let method_ty = self.db.value_ty(func.into()).substitute(Interner, &subst);
        self.register_obligations_for_call(&method_ty);

        self.infer_expr_coerce(rhs, &Expectation::has_type(rhs_ty.clone()));

        let ret_ty = match method_ty.callable_sig(self.db) {
            Some(sig) => {
                let p_left = &sig.params()[0];
                if matches!(op, BinaryOp::CmpOp(..) | BinaryOp::Assignment { .. }) {
                    if let &TyKind::Ref(mtbl, _, _) = p_left.kind(Interner) {
                        self.write_expr_adj(
                            lhs,
                            vec![Adjustment {
                                kind: Adjust::Borrow(AutoBorrow::Ref(mtbl)),
                                target: p_left.clone(),
                            }],
                        );
                    }
                }
                let p_right = &sig.params()[1];
                if matches!(op, BinaryOp::CmpOp(..)) {
                    if let &TyKind::Ref(mtbl, _, _) = p_right.kind(Interner) {
                        self.write_expr_adj(
                            rhs,
                            vec![Adjustment {
                                kind: Adjust::Borrow(AutoBorrow::Ref(mtbl)),
                                target: p_right.clone(),
                            }],
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
                            &Expectation::HasType(self.result.standard_types.never.clone()),
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
            } else if let Some(t) = expected.only_has_type(&mut self.table) {
                if self.coerce(Some(expr), &TyBuilder::unit(), &t).is_err() {
                    self.result.type_mismatches.insert(
                        expr.into(),
                        TypeMismatch { expected: t.clone(), actual: TyBuilder::unit() },
                    );
                }
                t
            } else {
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

        let resolved = method_resolution::lookup_method(
            self.db,
            &canonicalized_receiver.value,
            self.trait_env.clone(),
            &traits_in_scope,
            VisibleFromModule::Filter(self.resolver.module()),
            method_name,
        );
        let (receiver_ty, method_ty, substs) = match resolved {
            Some((adjust, func, visible)) => {
                let (ty, adjustments) = adjust.apply(&mut self.table, receiver_ty);
                let generics = generics(self.db.upcast(), func.into());
                let substs = self.substs_for_method_call(generics, generic_args);
                self.write_expr_adj(receiver, adjustments);
                self.write_method_resolution(tgt_expr, func, substs.clone());
                if !visible {
                    self.push_diagnostic(InferenceDiagnostic::PrivateAssocItem {
                        id: tgt_expr.into(),
                        item: func.into(),
                    })
                }
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
        let (formal_receiver_ty, param_tys, ret_ty, is_varargs) =
            match method_ty.callable_sig(self.db) {
                Some(sig) => {
                    if !sig.params().is_empty() {
                        (
                            sig.params()[0].clone(),
                            sig.params()[1..].to_vec(),
                            sig.ret().clone(),
                            sig.is_varargs,
                        )
                    } else {
                        (self.err_ty(), Vec::new(), sig.ret().clone(), sig.is_varargs)
                    }
                }
                None => (self.err_ty(), Vec::new(), self.err_ty(), true),
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

    fn check_call_arguments(
        &mut self,
        expr: ExprId,
        args: &[ExprId],
        expected_inputs: &[Ty],
        param_tys: &[Ty],
        skip_indices: &[u32],
        is_varargs: bool,
    ) {
        if args.len() != param_tys.len() + skip_indices.len() && !is_varargs {
            self.push_diagnostic(InferenceDiagnostic::MismatchedArgCount {
                call_expr: expr,
                expected: param_tys.len() + skip_indices.len(),
                found: args.len(),
            });
        }

        // Quoting https://github.com/rust-lang/rust/blob/6ef275e6c3cb1384ec78128eceeb4963ff788dca/src/librustc_typeck/check/mod.rs#L3325 --
        // We do this in a pretty awful way: first we type-check any arguments
        // that are not closures, then we type-check the closures. This is so
        // that we have more information about the types of arguments when we
        // type-check the functions. This isn't really the right way to do this.
        for check_closures in [false, true] {
            let mut skip_indices = skip_indices.into_iter().copied().fuse().peekable();
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

                while skip_indices.peek().map_or(false, |i| *i < idx as u32) {
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
                if !coercion_target.is_unknown()
                    && self.coerce(Some(arg), &ty, &coercion_target).is_err()
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
        def_generics: Generics,
        generic_args: Option<&GenericArgs>,
    ) -> Substitution {
        let (parent_params, self_params, type_params, const_params, impl_trait_params) =
            def_generics.provenance_split();
        assert_eq!(self_params, 0); // method shouldn't have another Self param
        let total_len = parent_params + type_params + const_params + impl_trait_params;
        let mut substs = Vec::with_capacity(total_len);

        // handle provided arguments
        if let Some(generic_args) = generic_args {
            // if args are provided, it should be all of them, but we can't rely on that
            for (arg, kind_id) in generic_args
                .args
                .iter()
                .filter(|arg| !matches!(arg, GenericArg::Lifetime(_)))
                .take(type_params + const_params)
                .zip(def_generics.iter_id())
            {
                if let Some(g) = generic_arg_to_chalk(
                    self.db,
                    kind_id,
                    arg,
                    self,
                    |this, type_ref| this.make_ty(type_ref),
                    |this, c, ty| {
                        const_or_path_to_chalk(
                            this.db,
                            &this.resolver,
                            ty,
                            c,
                            ParamLoweringMode::Placeholder,
                            || generics(this.db.upcast(), this.resolver.generic_def().unwrap()),
                            DebruijnIndex::INNERMOST,
                        )
                    },
                ) {
                    substs.push(g);
                }
            }
        };

        // Handle everything else as unknown. This also handles generic arguments for the method's
        // parent (impl or trait), which should come after those for the method.
        for (id, data) in def_generics.iter().skip(substs.len()) {
            match data {
                TypeOrConstParamData::TypeParamData(_) => {
                    substs.push(GenericArgData::Ty(self.table.new_type_var()).intern(Interner))
                }
                TypeOrConstParamData::ConstParamData(_) => {
                    substs.push(
                        GenericArgData::Const(self.table.new_const_var(
                            self.db.const_param_ty(ConstParamId::from_unchecked(id)),
                        ))
                        .intern(Interner),
                    )
                }
            }
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
                        let params_len = parameters.len(Interner);
                        let trait_params_len = generics(self.db.upcast(), trait_.into()).len();
                        let substs = Substitution::from_iter(
                            Interner,
                            // The generic parameters for the trait come after those for the
                            // function.
                            &parameters.as_slice(Interner)[params_len - trait_params_len..],
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

        let data = self.db.function_data(func);
        if data.legacy_const_generics_indices.is_empty() {
            return Default::default();
        }

        // only use legacy const generics if the param count matches with them
        if data.params.len() + data.legacy_const_generics_indices.len() != args.len() {
            if args.len() <= data.params.len() {
                return Default::default();
            } else {
                // there are more parameters than there should be without legacy
                // const params; use them
                let mut indices = data.legacy_const_generics_indices.clone();
                indices.sort();
                return indices;
            }
        }

        // check legacy const parameters
        for (subst_idx, arg_idx) in data.legacy_const_generics_indices.iter().copied().enumerate() {
            let arg = match subst.at(Interner, subst_idx).constant(Interner) {
                Some(c) => c,
                None => continue, // not a const parameter?
            };
            if arg_idx >= args.len() as u32 {
                continue;
            }
            let _ty = arg.data(Interner).ty.clone();
            let expected = Expectation::none(); // FIXME use actual const ty, when that is lowered correctly
            self.infer_expr(args[arg_idx as usize], &expected);
            // FIXME: evaluate and unify with the const
        }
        let mut indices = data.legacy_const_generics_indices.clone();
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

        if is_assign {
            self.result.standard_types.unit.clone()
        } else {
            output_ty
        }
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

    fn with_breakable_ctx<T>(
        &mut self,
        kind: BreakableKind,
        ty: Ty,
        label: Option<LabelId>,
        cb: impl FnOnce(&mut Self) -> T,
    ) -> (Option<Ty>, T) {
        self.breakables.push({
            let label = label.map(|label| self.body[label].name.clone());
            BreakableContext { kind, may_break: false, coerce: CoerceMany::new(ty), label }
        });
        let res = cb(self);
        let ctx = self.breakables.pop().expect("breakable stack broken");
        (ctx.may_break.then(|| ctx.coerce.complete()), res)
    }
}
