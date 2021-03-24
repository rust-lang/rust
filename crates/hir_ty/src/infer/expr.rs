//! Type inference for expressions.

use std::iter::{repeat, repeat_with};
use std::{mem, sync::Arc};

use chalk_ir::{cast::Cast, Mutability, TyVariableKind};
use hir_def::{
    expr::{Array, BinaryOp, Expr, ExprId, Literal, Statement, UnaryOp},
    path::{GenericArg, GenericArgs},
    resolver::resolver_for_expr,
    AssocContainerId, FieldId, Lookup,
};
use hir_expand::name::{name, Name};
use stdx::always;
use syntax::ast::RangeOp;

use crate::{
    autoderef,
    lower::lower_to_chalk_mutability,
    method_resolution, op,
    primitive::{self, UintTy},
    to_assoc_type_id, to_chalk_trait_id,
    traits::{chalk::from_chalk, FnTrait, InEnvironment},
    utils::{generics, variant_data, Generics},
    AdtId, Binders, CallableDefId, DomainGoal, FnPointer, FnSig, Interner, Rawness, Scalar,
    Substitution, TraitRef, Ty, TyKind,
};

use super::{
    find_breakable, BindingMode, BreakableContext, Diverges, Expectation, InferenceContext,
    InferenceDiagnostic, TypeMismatch,
};

impl<'a> InferenceContext<'a> {
    pub(super) fn infer_expr(&mut self, tgt_expr: ExprId, expected: &Expectation) -> Ty {
        let ty = self.infer_expr_inner(tgt_expr, expected);
        if ty.is_never() {
            // Any expression that produces a value of type `!` must have diverged
            self.diverges = Diverges::Always;
        }
        let could_unify = self.unify(&ty, &expected.ty);
        if !could_unify {
            self.result.type_mismatches.insert(
                tgt_expr,
                TypeMismatch { expected: expected.ty.clone(), actual: ty.clone() },
            );
        }
        self.resolve_ty_as_possible(ty)
    }

    /// Infer type of expression with possibly implicit coerce to the expected type.
    /// Return the type after possible coercion.
    pub(super) fn infer_expr_coerce(&mut self, expr: ExprId, expected: &Expectation) -> Ty {
        let ty = self.infer_expr_inner(expr, &expected);
        let ty = if !self.coerce(&ty, &expected.coercion_target()) {
            self.result
                .type_mismatches
                .insert(expr, TypeMismatch { expected: expected.ty.clone(), actual: ty.clone() });
            // Return actual type when type mismatch.
            // This is needed for diagnostic when return type mismatch.
            ty
        } else if expected.coercion_target().is_unknown() {
            ty
        } else {
            expected.ty.clone()
        };

        self.resolve_ty_as_possible(ty)
    }

    fn callable_sig_from_fn_trait(&mut self, ty: &Ty, num_args: usize) -> Option<(Vec<Ty>, Ty)> {
        let krate = self.resolver.krate()?;
        let fn_once_trait = FnTrait::FnOnce.get_id(self.db, krate)?;
        let output_assoc_type =
            self.db.trait_data(fn_once_trait).associated_type_by_name(&name![Output])?;
        let generic_params = generics(self.db.upcast(), fn_once_trait.into());
        if generic_params.len() != 2 {
            return None;
        }

        let mut param_builder = Substitution::builder(num_args);
        let mut arg_tys = vec![];
        for _ in 0..num_args {
            let arg = self.table.new_type_var();
            param_builder = param_builder.push(arg.clone());
            arg_tys.push(arg);
        }
        let parameters = param_builder.build();
        let arg_ty = TyKind::Tuple(num_args, parameters).intern(&Interner);
        let substs =
            Substitution::build_for_generics(&generic_params).push(ty.clone()).push(arg_ty).build();

        let trait_env = self.trait_env.env.clone();
        let implements_fn_trait: DomainGoal =
            TraitRef { trait_id: to_chalk_trait_id(fn_once_trait), substitution: substs.clone() }
                .cast(&Interner);
        let goal = self.canonicalizer().canonicalize_obligation(InEnvironment {
            goal: implements_fn_trait.clone(),
            environment: trait_env,
        });
        if self.db.trait_solve(krate, goal.value).is_some() {
            self.obligations.push(implements_fn_trait);
            let output_proj_ty = crate::ProjectionTy {
                associated_ty_id: to_assoc_type_id(output_assoc_type),
                substitution: substs,
            };
            let return_ty = self.normalize_projection_ty(output_proj_ty);
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
        let body = Arc::clone(&self.body); // avoid borrow checker problem
        let ty = match &body[tgt_expr] {
            Expr::Missing => self.err_ty(),
            Expr::If { condition, then_branch, else_branch } => {
                // if let is desugared to match, so this is always simple if
                self.infer_expr(
                    *condition,
                    &Expectation::has_type(TyKind::Scalar(Scalar::Bool).intern(&Interner)),
                );

                let condition_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
                let mut both_arms_diverge = Diverges::Always;

                let then_ty = self.infer_expr_inner(*then_branch, &expected);
                both_arms_diverge &= mem::replace(&mut self.diverges, Diverges::Maybe);
                let else_ty = match else_branch {
                    Some(else_branch) => self.infer_expr_inner(*else_branch, &expected),
                    None => Ty::unit(),
                };
                both_arms_diverge &= self.diverges;

                self.diverges = condition_diverges | both_arms_diverge;

                self.coerce_merge_branch(&then_ty, &else_ty)
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
                            break_ty: break_ty.clone(),
                            label: label.map(|label| self.body[label].name.clone()),
                        });
                        let ty =
                            self.infer_block(statements, *tail, &Expectation::has_type(break_ty));
                        let ctxt = self.breakables.pop().expect("breakable stack broken");
                        if ctxt.may_break {
                            ctxt.break_ty
                        } else {
                            ty
                        }
                    }
                    None => self.infer_block(statements, *tail, expected),
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
                // existenail type AsyncBlockImplTrait<InnerType>: Future<Output = InnerType>
                let inner_ty = self.infer_expr(*body, &Expectation::none());
                let impl_trait_id = crate::ImplTraitId::AsyncBlockTypeImplTrait(self.owner, *body);
                let opaque_ty_id = self.db.intern_impl_trait_id(impl_trait_id).into();
                TyKind::OpaqueType(opaque_ty_id, Substitution::single(inner_ty)).intern(&Interner)
            }
            Expr::Loop { body, label } => {
                self.breakables.push(BreakableContext {
                    may_break: false,
                    break_ty: self.table.new_type_var(),
                    label: label.map(|label| self.body[label].name.clone()),
                });
                self.infer_expr(*body, &Expectation::has_type(Ty::unit()));

                let ctxt = self.breakables.pop().expect("breakable stack broken");
                if ctxt.may_break {
                    self.diverges = Diverges::Maybe;
                }

                if ctxt.may_break {
                    ctxt.break_ty
                } else {
                    TyKind::Never.intern(&Interner)
                }
            }
            Expr::While { condition, body, label } => {
                self.breakables.push(BreakableContext {
                    may_break: false,
                    break_ty: self.err_ty(),
                    label: label.map(|label| self.body[label].name.clone()),
                });
                // while let is desugared to a match loop, so this is always simple while
                self.infer_expr(
                    *condition,
                    &Expectation::has_type(TyKind::Scalar(Scalar::Bool).intern(&Interner)),
                );
                self.infer_expr(*body, &Expectation::has_type(Ty::unit()));
                let _ctxt = self.breakables.pop().expect("breakable stack broken");
                // the body may not run, so it diverging doesn't mean we diverge
                self.diverges = Diverges::Maybe;
                Ty::unit()
            }
            Expr::For { iterable, body, pat, label } => {
                let iterable_ty = self.infer_expr(*iterable, &Expectation::none());

                self.breakables.push(BreakableContext {
                    may_break: false,
                    break_ty: self.err_ty(),
                    label: label.map(|label| self.body[label].name.clone()),
                });
                let pat_ty =
                    self.resolve_associated_type(iterable_ty, self.resolve_into_iter_item());

                self.infer_pat(*pat, &pat_ty, BindingMode::default());

                self.infer_expr(*body, &Expectation::has_type(Ty::unit()));
                let _ctxt = self.breakables.pop().expect("breakable stack broken");
                // the body may not run, so it diverging doesn't mean we diverge
                self.diverges = Diverges::Maybe;
                Ty::unit()
            }
            Expr::Lambda { body, args, ret_type, arg_types } => {
                assert_eq!(args.len(), arg_types.len());

                let mut sig_tys = Vec::new();

                // collect explicitly written argument types
                for arg_type in arg_types.iter() {
                    let arg_ty = if let Some(type_ref) = arg_type {
                        self.make_ty(type_ref)
                    } else {
                        self.table.new_type_var()
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
                    num_args: sig_tys.len() - 1,
                    sig: FnSig { abi: (), safety: chalk_ir::Safety::Safe, variadic: false },
                    substs: Substitution(sig_tys.clone().into()),
                })
                .intern(&Interner);
                let closure_id = self.db.intern_closure((self.owner, tgt_expr)).into();
                let closure_ty =
                    TyKind::Closure(closure_id, Substitution::single(sig_ty)).intern(&Interner);

                // Eagerly try to relate the closure type with the expected
                // type, otherwise we often won't have enough information to
                // infer the body.
                self.coerce(&closure_ty, &expected.ty);

                // Now go through the argument patterns
                for (arg_pat, arg_ty) in args.iter().zip(sig_tys) {
                    let resolved = self.resolve_ty_as_possible(arg_ty);
                    self.infer_pat(*arg_pat, &resolved, BindingMode::default());
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
                let canonicalized = self.canonicalizer().canonicalize_ty(callee_ty.clone());
                let mut derefs = autoderef(
                    self.db,
                    self.resolver.krate(),
                    InEnvironment {
                        goal: canonicalized.value.clone(),
                        environment: self.trait_env.env.clone(),
                    },
                );
                let (param_tys, ret_ty): (Vec<Ty>, Ty) = derefs
                    .find_map(|callee_deref_ty| {
                        self.callable_sig(
                            &canonicalized.decanonicalize_ty(callee_deref_ty.value),
                            args.len(),
                        )
                    })
                    .unwrap_or((Vec::new(), self.err_ty()));
                self.register_obligations_for_call(&callee_ty);
                self.check_call_arguments(args, &param_tys);
                self.normalize_associated_types_in(ret_ty)
            }
            Expr::MethodCall { receiver, args, method_name, generic_args } => self
                .infer_method_call(tgt_expr, *receiver, &args, &method_name, generic_args.as_ref()),
            Expr::Match { expr, arms } => {
                let input_ty = self.infer_expr(*expr, &Expectation::none());

                let mut result_ty = if arms.is_empty() {
                    TyKind::Never.intern(&Interner)
                } else {
                    self.table.new_type_var()
                };

                let matchee_diverges = self.diverges;
                let mut all_arms_diverge = Diverges::Always;

                for arm in arms {
                    self.diverges = Diverges::Maybe;
                    let _pat_ty = self.infer_pat(arm.pat, &input_ty, BindingMode::default());
                    if let Some(guard_expr) = arm.guard {
                        self.infer_expr(
                            guard_expr,
                            &Expectation::has_type(TyKind::Scalar(Scalar::Bool).intern(&Interner)),
                        );
                    }

                    let arm_ty = self.infer_expr_inner(arm.expr, &expected);
                    all_arms_diverge &= self.diverges;
                    result_ty = self.coerce_merge_branch(&result_ty, &arm_ty);
                }

                self.diverges = matchee_diverges | all_arms_diverge;

                result_ty
            }
            Expr::Path(p) => {
                // FIXME this could be more efficient...
                let resolver = resolver_for_expr(self.db.upcast(), self.owner, tgt_expr);
                self.infer_path(&resolver, p, tgt_expr.into()).unwrap_or(self.err_ty())
            }
            Expr::Continue { .. } => TyKind::Never.intern(&Interner),
            Expr::Break { expr, label } => {
                let val_ty = if let Some(expr) = expr {
                    self.infer_expr(*expr, &Expectation::none())
                } else {
                    Ty::unit()
                };

                let last_ty =
                    if let Some(ctxt) = find_breakable(&mut self.breakables, label.as_ref()) {
                        ctxt.break_ty.clone()
                    } else {
                        self.err_ty()
                    };

                let merged_type = self.coerce_merge_branch(&last_ty, &val_ty);

                if let Some(ctxt) = find_breakable(&mut self.breakables, label.as_ref()) {
                    ctxt.break_ty = merged_type;
                    ctxt.may_break = true;
                } else {
                    self.push_diagnostic(InferenceDiagnostic::BreakOutsideOfLoop {
                        expr: tgt_expr,
                    });
                }
                TyKind::Never.intern(&Interner)
            }
            Expr::Return { expr } => {
                if let Some(expr) = expr {
                    self.infer_expr_coerce(*expr, &Expectation::has_type(self.return_ty.clone()));
                } else {
                    let unit = Ty::unit();
                    self.coerce(&unit, &self.return_ty.clone());
                }
                TyKind::Never.intern(&Interner)
            }
            Expr::Yield { expr } => {
                // FIXME: track yield type for coercion
                if let Some(expr) = expr {
                    self.infer_expr(*expr, &Expectation::none());
                }
                TyKind::Never.intern(&Interner)
            }
            Expr::RecordLit { path, fields, spread } => {
                let (ty, def_id) = self.resolve_variant(path.as_ref());
                if let Some(variant) = def_id {
                    self.write_variant_resolution(tgt_expr.into(), variant);
                }

                self.unify(&ty, &expected.ty);

                let substs = ty.substs().cloned().unwrap_or_else(Substitution::empty);
                let field_types = def_id.map(|it| self.db.field_types(it)).unwrap_or_default();
                let variant_data = def_id.map(|it| variant_data(self.db.upcast(), it));
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
                    if let Some(field_def) = field_def {
                        self.result.record_field_resolutions.insert(field.expr, field_def);
                    }
                    let field_ty = field_def.map_or(self.err_ty(), |it| {
                        field_types[it.local_id].clone().subst(&substs)
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
                let canonicalized = self.canonicalizer().canonicalize_ty(receiver_ty);
                let ty = autoderef::autoderef(
                    self.db,
                    self.resolver.krate(),
                    InEnvironment {
                        goal: canonicalized.value.clone(),
                        environment: self.trait_env.env.clone(),
                    },
                )
                .find_map(|derefed_ty| {
                    let def_db = self.db.upcast();
                    let module = self.resolver.module();
                    let is_visible = |field_id: &FieldId| {
                        module
                            .map(|mod_id| {
                                self.db.field_visibilities(field_id.parent)[field_id.local_id]
                                    .is_visible_from(def_db, mod_id)
                            })
                            .unwrap_or(true)
                    };
                    match canonicalized.decanonicalize_ty(derefed_ty.value).interned(&Interner) {
                        TyKind::Tuple(_, substs) => {
                            name.as_tuple_index().and_then(|idx| substs.0.get(idx).cloned())
                        }
                        TyKind::Adt(AdtId(hir_def::AdtId::StructId(s)), parameters) => {
                            let local_id = self.db.struct_data(*s).variant_data.field(name)?;
                            let field = FieldId { parent: (*s).into(), local_id };
                            if is_visible(&field) {
                                self.write_field_resolution(tgt_expr, field);
                                Some(
                                    self.db.field_types((*s).into())[field.local_id]
                                        .clone()
                                        .subst(&parameters),
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
                                        .subst(&parameters),
                                )
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                })
                .unwrap_or(self.err_ty());
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
                let _inner_ty = self.infer_expr_inner(*expr, &Expectation::none());
                let cast_ty = self.make_ty(type_ref);
                // FIXME check the cast...
                cast_ty
            }
            Expr::Ref { expr, rawness, mutability } => {
                let mutability = lower_to_chalk_mutability(*mutability);
                let expectation = if let Some((exp_inner, exp_rawness, exp_mutability)) =
                    &expected.ty.as_reference_or_ptr()
                {
                    if *exp_mutability == Mutability::Mut && mutability == Mutability::Not {
                        // FIXME: throw type error - expected mut reference but found shared ref,
                        // which cannot be coerced
                    }
                    if *exp_rawness == Rawness::Ref && *rawness == Rawness::RawPtr {
                        // FIXME: throw type error - expected reference but found ptr,
                        // which cannot be coerced
                    }
                    Expectation::rvalue_hint(Ty::clone(exp_inner))
                } else {
                    Expectation::none()
                };
                let inner_ty = self.infer_expr_inner(*expr, &expectation);
                match rawness {
                    Rawness::RawPtr => TyKind::Raw(mutability, inner_ty),
                    Rawness::Ref => TyKind::Ref(mutability, inner_ty),
                }
                .intern(&Interner)
            }
            Expr::Box { expr } => {
                let inner_ty = self.infer_expr_inner(*expr, &Expectation::none());
                if let Some(box_) = self.resolve_boxed_box() {
                    let mut sb =
                        Substitution::build_for_generics(&generics(self.db.upcast(), box_.into()));
                    sb = sb.push(inner_ty);
                    match self.db.generic_defaults(box_.into()).get(1) {
                        Some(alloc_ty) if !alloc_ty.value.is_unknown() && sb.remaining() > 0 => {
                            sb = sb.push(alloc_ty.value.clone());
                        }
                        _ => (),
                    }
                    sb = sb.fill(repeat_with(|| self.table.new_type_var()));
                    Ty::adt_ty(box_, sb.build())
                } else {
                    self.err_ty()
                }
            }
            Expr::UnaryOp { expr, op } => {
                let inner_ty = self.infer_expr_inner(*expr, &Expectation::none());
                match op {
                    UnaryOp::Deref => match self.resolver.krate() {
                        Some(krate) => {
                            let canonicalized = self.canonicalizer().canonicalize_ty(inner_ty);
                            match autoderef::deref(
                                self.db,
                                krate,
                                InEnvironment {
                                    goal: &canonicalized.value,
                                    environment: self.trait_env.env.clone(),
                                },
                            ) {
                                Some(derefed_ty) => {
                                    canonicalized.decanonicalize_ty(derefed_ty.value)
                                }
                                None => self.err_ty(),
                            }
                        }
                        None => self.err_ty(),
                    },
                    UnaryOp::Neg => {
                        match inner_ty.interned(&Interner) {
                            // Fast path for builtins
                            TyKind::Scalar(Scalar::Int(_))
                            | TyKind::Scalar(Scalar::Uint(_))
                            | TyKind::Scalar(Scalar::Float(_))
                            | TyKind::InferenceVar(_, TyVariableKind::Integer)
                            | TyKind::InferenceVar(_, TyVariableKind::Float) => inner_ty,
                            // Otherwise we resolve via the std::ops::Neg trait
                            _ => self
                                .resolve_associated_type(inner_ty, self.resolve_ops_neg_output()),
                        }
                    }
                    UnaryOp::Not => {
                        match inner_ty.interned(&Interner) {
                            // Fast path for builtins
                            TyKind::Scalar(Scalar::Bool)
                            | TyKind::Scalar(Scalar::Int(_))
                            | TyKind::Scalar(Scalar::Uint(_))
                            | TyKind::InferenceVar(_, TyVariableKind::Integer) => inner_ty,
                            // Otherwise we resolve via the std::ops::Not trait
                            _ => self
                                .resolve_associated_type(inner_ty, self.resolve_ops_not_output()),
                        }
                    }
                }
            }
            Expr::BinaryOp { lhs, rhs, op } => match op {
                Some(op) => {
                    let lhs_expectation = match op {
                        BinaryOp::LogicOp(..) => {
                            Expectation::has_type(TyKind::Scalar(Scalar::Bool).intern(&Interner))
                        }
                        _ => Expectation::none(),
                    };
                    let lhs_ty = self.infer_expr(*lhs, &lhs_expectation);
                    let rhs_expectation = op::binary_op_rhs_expectation(*op, lhs_ty.clone());
                    let rhs_ty = self.infer_expr(*rhs, &Expectation::has_type(rhs_expectation));

                    let ret = op::binary_op_return_ty(*op, lhs_ty.clone(), rhs_ty.clone());

                    if ret.is_unknown() {
                        cov_mark::hit!(infer_expr_inner_binary_operator_overload);

                        self.resolve_associated_type_with_params(
                            lhs_ty,
                            self.resolve_binary_op_output(op),
                            &[rhs_ty],
                        )
                    } else {
                        ret
                    }
                }
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
                        Some(adt) => Ty::adt_ty(adt, Substitution::empty()),
                        None => self.err_ty(),
                    },
                    (RangeOp::Exclusive, None, Some(ty)) => match self.resolve_range_to() {
                        Some(adt) => Ty::adt_ty(adt, Substitution::single(ty)),
                        None => self.err_ty(),
                    },
                    (RangeOp::Inclusive, None, Some(ty)) => {
                        match self.resolve_range_to_inclusive() {
                            Some(adt) => Ty::adt_ty(adt, Substitution::single(ty)),
                            None => self.err_ty(),
                        }
                    }
                    (RangeOp::Exclusive, Some(_), Some(ty)) => match self.resolve_range() {
                        Some(adt) => Ty::adt_ty(adt, Substitution::single(ty)),
                        None => self.err_ty(),
                    },
                    (RangeOp::Inclusive, Some(_), Some(ty)) => {
                        match self.resolve_range_inclusive() {
                            Some(adt) => Ty::adt_ty(adt, Substitution::single(ty)),
                            None => self.err_ty(),
                        }
                    }
                    (RangeOp::Exclusive, Some(ty), None) => match self.resolve_range_from() {
                        Some(adt) => Ty::adt_ty(adt, Substitution::single(ty)),
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
                    let canonicalized = self.canonicalizer().canonicalize_ty(base_ty);
                    let self_ty = method_resolution::resolve_indexing_op(
                        self.db,
                        &canonicalized.value,
                        self.trait_env.clone(),
                        krate,
                        index_trait,
                    );
                    let self_ty =
                        self_ty.map_or(self.err_ty(), |t| canonicalized.decanonicalize_ty(t.value));
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
                let mut tys = match expected.ty.interned(&Interner) {
                    TyKind::Tuple(_, substs) => substs
                        .iter()
                        .cloned()
                        .chain(repeat_with(|| self.table.new_type_var()))
                        .take(exprs.len())
                        .collect::<Vec<_>>(),
                    _ => (0..exprs.len()).map(|_| self.table.new_type_var()).collect(),
                };

                for (expr, ty) in exprs.iter().zip(tys.iter_mut()) {
                    self.infer_expr_coerce(*expr, &Expectation::has_type(ty.clone()));
                }

                TyKind::Tuple(tys.len(), Substitution(tys.into())).intern(&Interner)
            }
            Expr::Array(array) => {
                let elem_ty = match expected.ty.interned(&Interner) {
                    TyKind::Array(st) | TyKind::Slice(st) => st.clone(),
                    _ => self.table.new_type_var(),
                };

                match array {
                    Array::ElementList(items) => {
                        for expr in items.iter() {
                            self.infer_expr_coerce(*expr, &Expectation::has_type(elem_ty.clone()));
                        }
                    }
                    Array::Repeat { initializer, repeat } => {
                        self.infer_expr_coerce(
                            *initializer,
                            &Expectation::has_type(elem_ty.clone()),
                        );
                        self.infer_expr(
                            *repeat,
                            &Expectation::has_type(
                                TyKind::Scalar(Scalar::Uint(UintTy::Usize)).intern(&Interner),
                            ),
                        );
                    }
                }

                TyKind::Array(elem_ty).intern(&Interner)
            }
            Expr::Literal(lit) => match lit {
                Literal::Bool(..) => TyKind::Scalar(Scalar::Bool).intern(&Interner),
                Literal::String(..) => {
                    TyKind::Ref(Mutability::Not, TyKind::Str.intern(&Interner)).intern(&Interner)
                }
                Literal::ByteString(..) => {
                    let byte_type = TyKind::Scalar(Scalar::Uint(UintTy::U8)).intern(&Interner);
                    let array_type = TyKind::Array(byte_type).intern(&Interner);
                    TyKind::Ref(Mutability::Not, array_type).intern(&Interner)
                }
                Literal::Char(..) => TyKind::Scalar(Scalar::Char).intern(&Interner),
                Literal::Int(_v, ty) => match ty {
                    Some(int_ty) => {
                        TyKind::Scalar(Scalar::Int(primitive::int_ty_from_builtin(*int_ty)))
                            .intern(&Interner)
                    }
                    None => self.table.new_integer_var(),
                },
                Literal::Uint(_v, ty) => match ty {
                    Some(int_ty) => {
                        TyKind::Scalar(Scalar::Uint(primitive::uint_ty_from_builtin(*int_ty)))
                            .intern(&Interner)
                    }
                    None => self.table.new_integer_var(),
                },
                Literal::Float(_v, ty) => match ty {
                    Some(float_ty) => {
                        TyKind::Scalar(Scalar::Float(primitive::float_ty_from_builtin(*float_ty)))
                            .intern(&Interner)
                    }
                    None => self.table.new_float_var(),
                },
            },
        };
        // use a new type variable if we got unknown here
        let ty = self.insert_type_vars_shallow(ty);
        let ty = self.resolve_ty_as_possible(ty);
        self.write_expr_ty(tgt_expr, ty.clone());
        ty
    }

    fn infer_block(
        &mut self,
        statements: &[Statement],
        tail: Option<ExprId>,
        expected: &Expectation,
    ) -> Ty {
        for stmt in statements {
            match stmt {
                Statement::Let { pat, type_ref, initializer } => {
                    let decl_ty =
                        type_ref.as_ref().map(|tr| self.make_ty(tr)).unwrap_or(self.err_ty());

                    // Always use the declared type when specified
                    let mut ty = decl_ty.clone();

                    if let Some(expr) = initializer {
                        let actual_ty =
                            self.infer_expr_coerce(*expr, &Expectation::has_type(decl_ty.clone()));
                        if decl_ty.is_unknown() {
                            ty = actual_ty;
                        }
                    }

                    let ty = self.resolve_ty_as_possible(ty);
                    self.infer_pat(*pat, &ty, BindingMode::default());
                }
                Statement::Expr(expr) => {
                    self.infer_expr(*expr, &Expectation::none());
                }
            }
        }

        let ty = if let Some(expr) = tail {
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
                self.coerce(&Ty::unit(), &expected.coercion_target());
                Ty::unit()
            }
        };
        ty
    }

    fn infer_method_call(
        &mut self,
        tgt_expr: ExprId,
        receiver: ExprId,
        args: &[ExprId],
        method_name: &Name,
        generic_args: Option<&GenericArgs>,
    ) -> Ty {
        let receiver_ty = self.infer_expr(receiver, &Expectation::none());
        let canonicalized_receiver = self.canonicalizer().canonicalize_ty(receiver_ty.clone());

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
        let (derefed_receiver_ty, method_ty, def_generics) = match resolved {
            Some((ty, func)) => {
                let ty = canonicalized_receiver.decanonicalize_ty(ty);
                self.write_method_resolution(tgt_expr, func);
                (ty, self.db.value_ty(func.into()), Some(generics(self.db.upcast(), func.into())))
            }
            None => (receiver_ty, Binders::new(0, self.err_ty()), None),
        };
        let substs = self.substs_for_method_call(def_generics, generic_args, &derefed_receiver_ty);
        let method_ty = method_ty.subst(&substs);
        let method_ty = self.insert_type_vars(method_ty);
        self.register_obligations_for_call(&method_ty);
        let (expected_receiver_ty, param_tys, ret_ty) = match method_ty.callable_sig(self.db) {
            Some(sig) => {
                if !sig.params().is_empty() {
                    (sig.params()[0].clone(), sig.params()[1..].to_vec(), sig.ret().clone())
                } else {
                    (self.err_ty(), Vec::new(), sig.ret().clone())
                }
            }
            None => (self.err_ty(), Vec::new(), self.err_ty()),
        };
        // Apply autoref so the below unification works correctly
        // FIXME: return correct autorefs from lookup_method
        let actual_receiver_ty = match expected_receiver_ty.as_reference() {
            Some((_, mutability)) => TyKind::Ref(mutability, derefed_receiver_ty).intern(&Interner),
            _ => derefed_receiver_ty,
        };
        self.unify(&expected_receiver_ty, &actual_receiver_ty);

        self.check_call_arguments(args, &param_tys);
        self.normalize_associated_types_in(ret_ty)
    }

    fn check_call_arguments(&mut self, args: &[ExprId], param_tys: &[Ty]) {
        // Quoting https://github.com/rust-lang/rust/blob/6ef275e6c3cb1384ec78128eceeb4963ff788dca/src/librustc_typeck/check/mod.rs#L3325 --
        // We do this in a pretty awful way: first we type-check any arguments
        // that are not closures, then we type-check the closures. This is so
        // that we have more information about the types of arguments when we
        // type-check the functions. This isn't really the right way to do this.
        for &check_closures in &[false, true] {
            let param_iter = param_tys.iter().cloned().chain(repeat(self.err_ty()));
            for (&arg, param_ty) in args.iter().zip(param_iter) {
                let is_closure = matches!(&self.body[arg], Expr::Lambda { .. });
                if is_closure != check_closures {
                    continue;
                }

                let param_ty = self.normalize_associated_types_in(param_ty);
                self.infer_expr_coerce(arg, &Expectation::has_type(param_ty.clone()));
            }
        }
    }

    fn substs_for_method_call(
        &mut self,
        def_generics: Option<Generics>,
        generic_args: Option<&GenericArgs>,
        receiver_ty: &Ty,
    ) -> Substitution {
        let (parent_params, self_params, type_params, impl_trait_params) =
            def_generics.as_ref().map_or((0, 0, 0, 0), |g| g.provenance_split());
        assert_eq!(self_params, 0); // method shouldn't have another Self param
        let total_len = parent_params + type_params + impl_trait_params;
        let mut substs = Vec::with_capacity(total_len);
        // Parent arguments are unknown, except for the receiver type
        if let Some(parent_generics) = def_generics.as_ref().map(|p| p.iter_parent()) {
            for (_id, param) in parent_generics {
                if param.provenance == hir_def::generics::TypeParamProvenance::TraitSelf {
                    substs.push(receiver_ty.clone());
                } else {
                    substs.push(self.err_ty());
                }
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
            substs.push(self.err_ty());
        }
        assert_eq!(substs.len(), total_len);
        Substitution(substs.into())
    }

    fn register_obligations_for_call(&mut self, callable_ty: &Ty) {
        if let TyKind::FnDef(fn_def, parameters) = callable_ty.interned(&Interner) {
            let def: CallableDefId = from_chalk(self.db, *fn_def);
            let generic_predicates = self.db.generic_predicates(def.into());
            for predicate in generic_predicates.iter() {
                let (predicate, binders) =
                    predicate.clone().subst(parameters).into_value_and_skipped_binders();
                always!(binders == 0); // quantified where clauses not yet handled
                self.obligations.push(predicate.cast(&Interner));
            }
            // add obligation for trait implementation, if this is a trait method
            match def {
                CallableDefId::FunctionId(f) => {
                    if let AssocContainerId::TraitId(trait_) = f.lookup(self.db.upcast()).container
                    {
                        // construct a TraitRef
                        let substs =
                            parameters.prefix(generics(self.db.upcast(), trait_.into()).len());
                        self.obligations.push(
                            TraitRef { trait_id: to_chalk_trait_id(trait_), substitution: substs }
                                .cast(&Interner),
                        );
                    }
                }
                CallableDefId::StructId(_) | CallableDefId::EnumVariantId(_) => {}
            }
        }
    }
}
