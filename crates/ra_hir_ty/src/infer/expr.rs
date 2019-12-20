//! Type inference for expressions.

use std::iter::{repeat, repeat_with};
use std::sync::Arc;

use hir_def::{
    builtin_type::Signedness,
    expr::{Array, BinaryOp, Expr, ExprId, Literal, Statement, UnaryOp},
    path::{GenericArg, GenericArgs},
    resolver::resolver_for_expr,
    AdtId, AssocContainerId, Lookup, StructFieldId,
};
use hir_expand::name::{name, Name};
use ra_syntax::ast::RangeOp;

use crate::{
    autoderef,
    db::HirDatabase,
    method_resolution, op,
    traits::InEnvironment,
    utils::{generics, variant_data, Generics},
    ApplicationTy, CallableDef, InferTy, IntTy, Mutability, Obligation, Substs, TraitRef, Ty,
    TypeCtor, TypeWalk, Uncertain,
};

use super::{BindingMode, Expectation, InferenceContext, InferenceDiagnostic, TypeMismatch};

impl<'a, D: HirDatabase> InferenceContext<'a, D> {
    pub(super) fn infer_expr(&mut self, tgt_expr: ExprId, expected: &Expectation) -> Ty {
        let ty = self.infer_expr_inner(tgt_expr, expected);
        let could_unify = self.unify(&ty, &expected.ty);
        if !could_unify {
            self.result.type_mismatches.insert(
                tgt_expr,
                TypeMismatch { expected: expected.ty.clone(), actual: ty.clone() },
            );
        }
        let ty = self.resolve_ty_as_possible(ty);
        ty
    }

    /// Infer type of expression with possibly implicit coerce to the expected type.
    /// Return the type after possible coercion.
    pub(super) fn infer_expr_coerce(&mut self, expr: ExprId, expected: &Expectation) -> Ty {
        let ty = self.infer_expr_inner(expr, &expected);
        let ty = if !self.coerce(&ty, &expected.ty) {
            self.result
                .type_mismatches
                .insert(expr, TypeMismatch { expected: expected.ty.clone(), actual: ty.clone() });
            // Return actual type when type mismatch.
            // This is needed for diagnostic when return type mismatch.
            ty
        } else if expected.ty == Ty::Unknown {
            ty
        } else {
            expected.ty.clone()
        };

        self.resolve_ty_as_possible(ty)
    }

    fn infer_expr_inner(&mut self, tgt_expr: ExprId, expected: &Expectation) -> Ty {
        let body = Arc::clone(&self.body); // avoid borrow checker problem
        let ty = match &body[tgt_expr] {
            Expr::Missing => Ty::Unknown,
            Expr::If { condition, then_branch, else_branch } => {
                // if let is desugared to match, so this is always simple if
                self.infer_expr(*condition, &Expectation::has_type(Ty::simple(TypeCtor::Bool)));

                let then_ty = self.infer_expr_inner(*then_branch, &expected);
                let else_ty = match else_branch {
                    Some(else_branch) => self.infer_expr_inner(*else_branch, &expected),
                    None => Ty::unit(),
                };

                self.coerce_merge_branch(&then_ty, &else_ty)
            }
            Expr::Block { statements, tail } => self.infer_block(statements, *tail, expected),
            Expr::TryBlock { body } => {
                let _inner = self.infer_expr(*body, expected);
                // FIXME should be std::result::Result<{inner}, _>
                Ty::Unknown
            }
            Expr::Loop { body } => {
                self.infer_expr(*body, &Expectation::has_type(Ty::unit()));
                // FIXME handle break with value
                Ty::simple(TypeCtor::Never)
            }
            Expr::While { condition, body } => {
                // while let is desugared to a match loop, so this is always simple while
                self.infer_expr(*condition, &Expectation::has_type(Ty::simple(TypeCtor::Bool)));
                self.infer_expr(*body, &Expectation::has_type(Ty::unit()));
                Ty::unit()
            }
            Expr::For { iterable, body, pat } => {
                let iterable_ty = self.infer_expr(*iterable, &Expectation::none());

                let pat_ty =
                    self.resolve_associated_type(iterable_ty, self.resolve_into_iter_item());

                self.infer_pat(*pat, &pat_ty, BindingMode::default());
                self.infer_expr(*body, &Expectation::has_type(Ty::unit()));
                Ty::unit()
            }
            Expr::Lambda { body, args, ret_type, arg_types } => {
                assert_eq!(args.len(), arg_types.len());

                let mut sig_tys = Vec::new();

                for (arg_pat, arg_type) in args.iter().zip(arg_types.iter()) {
                    let expected = if let Some(type_ref) = arg_type {
                        self.make_ty(type_ref)
                    } else {
                        Ty::Unknown
                    };
                    let arg_ty = self.infer_pat(*arg_pat, &expected, BindingMode::default());
                    sig_tys.push(arg_ty);
                }

                // add return type
                let ret_ty = match ret_type {
                    Some(type_ref) => self.make_ty(type_ref),
                    None => self.table.new_type_var(),
                };
                sig_tys.push(ret_ty.clone());
                let sig_ty = Ty::apply(
                    TypeCtor::FnPtr { num_args: sig_tys.len() as u16 - 1 },
                    Substs(sig_tys.into()),
                );
                let closure_ty = Ty::apply_one(
                    TypeCtor::Closure { def: self.owner.into(), expr: tgt_expr },
                    sig_ty,
                );

                // Eagerly try to relate the closure type with the expected
                // type, otherwise we often won't have enough information to
                // infer the body.
                self.coerce(&closure_ty, &expected.ty);

                let prev_ret_ty = std::mem::replace(&mut self.return_ty, ret_ty.clone());

                self.infer_expr_coerce(*body, &Expectation::has_type(ret_ty));

                self.return_ty = prev_ret_ty;

                closure_ty
            }
            Expr::Call { callee, args } => {
                let callee_ty = self.infer_expr(*callee, &Expectation::none());
                let (param_tys, ret_ty) = match callee_ty.callable_sig(self.db) {
                    Some(sig) => (sig.params().to_vec(), sig.ret().clone()),
                    None => {
                        // Not callable
                        // FIXME: report an error
                        (Vec::new(), Ty::Unknown)
                    }
                };
                self.register_obligations_for_call(&callee_ty);
                self.check_call_arguments(args, &param_tys);
                let ret_ty = self.normalize_associated_types_in(ret_ty);
                ret_ty
            }
            Expr::MethodCall { receiver, args, method_name, generic_args } => self
                .infer_method_call(tgt_expr, *receiver, &args, &method_name, generic_args.as_ref()),
            Expr::Match { expr, arms } => {
                let input_ty = self.infer_expr(*expr, &Expectation::none());

                let mut result_ty = self.table.new_maybe_never_type_var();

                for arm in arms {
                    for &pat in &arm.pats {
                        let _pat_ty = self.infer_pat(pat, &input_ty, BindingMode::default());
                    }
                    if let Some(guard_expr) = arm.guard {
                        self.infer_expr(
                            guard_expr,
                            &Expectation::has_type(Ty::simple(TypeCtor::Bool)),
                        );
                    }

                    let arm_ty = self.infer_expr_inner(arm.expr, &expected);
                    result_ty = self.coerce_merge_branch(&result_ty, &arm_ty);
                }

                result_ty
            }
            Expr::Path(p) => {
                // FIXME this could be more efficient...
                let resolver = resolver_for_expr(self.db, self.owner.into(), tgt_expr);
                self.infer_path(&resolver, p, tgt_expr.into()).unwrap_or(Ty::Unknown)
            }
            Expr::Continue => Ty::simple(TypeCtor::Never),
            Expr::Break { expr } => {
                if let Some(expr) = expr {
                    // FIXME handle break with value
                    self.infer_expr(*expr, &Expectation::none());
                }
                Ty::simple(TypeCtor::Never)
            }
            Expr::Return { expr } => {
                if let Some(expr) = expr {
                    self.infer_expr_coerce(*expr, &Expectation::has_type(self.return_ty.clone()));
                } else {
                    let unit = Ty::unit();
                    self.coerce(&unit, &self.return_ty.clone());
                }
                Ty::simple(TypeCtor::Never)
            }
            Expr::RecordLit { path, fields, spread } => {
                let (ty, def_id) = self.resolve_variant(path.as_ref());
                if let Some(variant) = def_id {
                    self.write_variant_resolution(tgt_expr.into(), variant);
                }

                self.unify(&ty, &expected.ty);

                let substs = ty.substs().unwrap_or_else(Substs::empty);
                let field_types =
                    def_id.map(|it| self.db.field_types(it.into())).unwrap_or_default();
                let variant_data = def_id.map(|it| variant_data(self.db, it));
                for (field_idx, field) in fields.iter().enumerate() {
                    let field_def =
                        variant_data.as_ref().and_then(|it| match it.field(&field.name) {
                            Some(local_id) => {
                                Some(StructFieldId { parent: def_id.unwrap(), local_id })
                            }
                            None => {
                                self.push_diagnostic(InferenceDiagnostic::NoSuchField {
                                    expr: tgt_expr,
                                    field: field_idx,
                                });
                                None
                            }
                        });
                    if let Some(field_def) = field_def {
                        self.result.record_field_resolutions.insert(field.expr, field_def);
                    }
                    let field_ty = field_def
                        .map_or(Ty::Unknown, |it| field_types[it.local_id].clone())
                        .subst(&substs);
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
                        value: canonicalized.value.clone(),
                        environment: self.trait_env.clone(),
                    },
                )
                .find_map(|derefed_ty| match canonicalized.decanonicalize_ty(derefed_ty.value) {
                    Ty::Apply(a_ty) => match a_ty.ctor {
                        TypeCtor::Tuple { .. } => name
                            .as_tuple_index()
                            .and_then(|idx| a_ty.parameters.0.get(idx).cloned()),
                        TypeCtor::Adt(AdtId::StructId(s)) => {
                            self.db.struct_data(s).variant_data.field(name).map(|local_id| {
                                let field = StructFieldId { parent: s.into(), local_id }.into();
                                self.write_field_resolution(tgt_expr, field);
                                self.db.field_types(s.into())[field.local_id]
                                    .clone()
                                    .subst(&a_ty.parameters)
                            })
                        }
                        // FIXME:
                        TypeCtor::Adt(AdtId::UnionId(_)) => None,
                        _ => None,
                    },
                    _ => None,
                })
                .unwrap_or(Ty::Unknown);
                let ty = self.insert_type_vars(ty);
                self.normalize_associated_types_in(ty)
            }
            Expr::Await { expr } => {
                let inner_ty = self.infer_expr_inner(*expr, &Expectation::none());
                let ty =
                    self.resolve_associated_type(inner_ty, self.resolve_future_future_output());
                ty
            }
            Expr::Try { expr } => {
                let inner_ty = self.infer_expr_inner(*expr, &Expectation::none());
                let ty = self.resolve_associated_type(inner_ty, self.resolve_ops_try_ok());
                ty
            }
            Expr::Cast { expr, type_ref } => {
                let _inner_ty = self.infer_expr_inner(*expr, &Expectation::none());
                let cast_ty = self.make_ty(type_ref);
                // FIXME check the cast...
                cast_ty
            }
            Expr::Ref { expr, mutability } => {
                let expectation =
                    if let Some((exp_inner, exp_mutability)) = &expected.ty.as_reference() {
                        if *exp_mutability == Mutability::Mut && *mutability == Mutability::Shared {
                            // FIXME: throw type error - expected mut reference but found shared ref,
                            // which cannot be coerced
                        }
                        Expectation::has_type(Ty::clone(exp_inner))
                    } else {
                        Expectation::none()
                    };
                let inner_ty = self.infer_expr_inner(*expr, &expectation);
                Ty::apply_one(TypeCtor::Ref(*mutability), inner_ty)
            }
            Expr::Box { expr } => {
                let inner_ty = self.infer_expr_inner(*expr, &Expectation::none());
                if let Some(box_) = self.resolve_boxed_box() {
                    Ty::apply_one(TypeCtor::Adt(box_), inner_ty)
                } else {
                    Ty::Unknown
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
                                    value: &canonicalized.value,
                                    environment: self.trait_env.clone(),
                                },
                            ) {
                                Some(derefed_ty) => {
                                    canonicalized.decanonicalize_ty(derefed_ty.value)
                                }
                                None => Ty::Unknown,
                            }
                        }
                        None => Ty::Unknown,
                    },
                    UnaryOp::Neg => {
                        match &inner_ty {
                            // Fast path for builtins
                            Ty::Apply(ApplicationTy {
                                ctor:
                                    TypeCtor::Int(Uncertain::Known(IntTy {
                                        signedness: Signedness::Signed,
                                        ..
                                    })),
                                ..
                            })
                            | Ty::Apply(ApplicationTy {
                                ctor: TypeCtor::Int(Uncertain::Unknown),
                                ..
                            })
                            | Ty::Apply(ApplicationTy { ctor: TypeCtor::Float(_), .. })
                            | Ty::Infer(InferTy::IntVar(..))
                            | Ty::Infer(InferTy::FloatVar(..)) => inner_ty,
                            // Otherwise we resolve via the std::ops::Neg trait
                            _ => self
                                .resolve_associated_type(inner_ty, self.resolve_ops_neg_output()),
                        }
                    }
                    UnaryOp::Not => {
                        match &inner_ty {
                            // Fast path for builtins
                            Ty::Apply(ApplicationTy { ctor: TypeCtor::Bool, .. })
                            | Ty::Apply(ApplicationTy { ctor: TypeCtor::Int(_), .. })
                            | Ty::Infer(InferTy::IntVar(..)) => inner_ty,
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
                        BinaryOp::LogicOp(..) => Expectation::has_type(Ty::simple(TypeCtor::Bool)),
                        _ => Expectation::none(),
                    };
                    let lhs_ty = self.infer_expr(*lhs, &lhs_expectation);
                    // FIXME: find implementation of trait corresponding to operation
                    // symbol and resolve associated `Output` type
                    let rhs_expectation = op::binary_op_rhs_expectation(*op, lhs_ty);
                    let rhs_ty = self.infer_expr(*rhs, &Expectation::has_type(rhs_expectation));

                    // FIXME: similar as above, return ty is often associated trait type
                    op::binary_op_return_ty(*op, rhs_ty)
                }
                _ => Ty::Unknown,
            },
            Expr::Range { lhs, rhs, range_type } => {
                let lhs_ty = lhs.map(|e| self.infer_expr_inner(e, &Expectation::none()));
                let rhs_expect = lhs_ty
                    .as_ref()
                    .map_or_else(Expectation::none, |ty| Expectation::has_type(ty.clone()));
                let rhs_ty = rhs.map(|e| self.infer_expr(e, &rhs_expect));
                match (range_type, lhs_ty, rhs_ty) {
                    (RangeOp::Exclusive, None, None) => match self.resolve_range_full() {
                        Some(adt) => Ty::simple(TypeCtor::Adt(adt)),
                        None => Ty::Unknown,
                    },
                    (RangeOp::Exclusive, None, Some(ty)) => match self.resolve_range_to() {
                        Some(adt) => Ty::apply_one(TypeCtor::Adt(adt), ty),
                        None => Ty::Unknown,
                    },
                    (RangeOp::Inclusive, None, Some(ty)) => {
                        match self.resolve_range_to_inclusive() {
                            Some(adt) => Ty::apply_one(TypeCtor::Adt(adt), ty),
                            None => Ty::Unknown,
                        }
                    }
                    (RangeOp::Exclusive, Some(_), Some(ty)) => match self.resolve_range() {
                        Some(adt) => Ty::apply_one(TypeCtor::Adt(adt), ty),
                        None => Ty::Unknown,
                    },
                    (RangeOp::Inclusive, Some(_), Some(ty)) => {
                        match self.resolve_range_inclusive() {
                            Some(adt) => Ty::apply_one(TypeCtor::Adt(adt), ty),
                            None => Ty::Unknown,
                        }
                    }
                    (RangeOp::Exclusive, Some(ty), None) => match self.resolve_range_from() {
                        Some(adt) => Ty::apply_one(TypeCtor::Adt(adt), ty),
                        None => Ty::Unknown,
                    },
                    (RangeOp::Inclusive, _, None) => Ty::Unknown,
                }
            }
            Expr::Index { base, index } => {
                let base_ty = self.infer_expr_inner(*base, &Expectation::none());
                let index_ty = self.infer_expr(*index, &Expectation::none());

                self.resolve_associated_type_with_params(
                    base_ty,
                    self.resolve_ops_index_output(),
                    &[index_ty],
                )
            }
            Expr::Tuple { exprs } => {
                let mut tys = match &expected.ty {
                    ty_app!(TypeCtor::Tuple { .. }, st) => st
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

                Ty::apply(TypeCtor::Tuple { cardinality: tys.len() as u16 }, Substs(tys.into()))
            }
            Expr::Array(array) => {
                let elem_ty = match &expected.ty {
                    ty_app!(TypeCtor::Array, st) | ty_app!(TypeCtor::Slice, st) => {
                        st.as_single().clone()
                    }
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
                            &Expectation::has_type(Ty::simple(TypeCtor::Int(Uncertain::Known(
                                IntTy::usize(),
                            )))),
                        );
                    }
                }

                Ty::apply_one(TypeCtor::Array, elem_ty)
            }
            Expr::Literal(lit) => match lit {
                Literal::Bool(..) => Ty::simple(TypeCtor::Bool),
                Literal::String(..) => {
                    Ty::apply_one(TypeCtor::Ref(Mutability::Shared), Ty::simple(TypeCtor::Str))
                }
                Literal::ByteString(..) => {
                    let byte_type = Ty::simple(TypeCtor::Int(Uncertain::Known(IntTy::u8())));
                    let slice_type = Ty::apply_one(TypeCtor::Slice, byte_type);
                    Ty::apply_one(TypeCtor::Ref(Mutability::Shared), slice_type)
                }
                Literal::Char(..) => Ty::simple(TypeCtor::Char),
                Literal::Int(_v, ty) => Ty::simple(TypeCtor::Int((*ty).into())),
                Literal::Float(_v, ty) => Ty::simple(TypeCtor::Float((*ty).into())),
            },
        };
        // use a new type variable if we got Ty::Unknown here
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
        let mut diverges = false;
        for stmt in statements {
            match stmt {
                Statement::Let { pat, type_ref, initializer } => {
                    let decl_ty =
                        type_ref.as_ref().map(|tr| self.make_ty(tr)).unwrap_or(Ty::Unknown);

                    // Always use the declared type when specified
                    let mut ty = decl_ty.clone();

                    if let Some(expr) = initializer {
                        let actual_ty =
                            self.infer_expr_coerce(*expr, &Expectation::has_type(decl_ty.clone()));
                        if decl_ty == Ty::Unknown {
                            ty = actual_ty;
                        }
                    }

                    let ty = self.resolve_ty_as_possible(ty);
                    self.infer_pat(*pat, &ty, BindingMode::default());
                }
                Statement::Expr(expr) => {
                    if let ty_app!(TypeCtor::Never) = self.infer_expr(*expr, &Expectation::none()) {
                        diverges = true;
                    }
                }
            }
        }

        let ty = if let Some(expr) = tail {
            self.infer_expr_coerce(expr, expected)
        } else {
            self.coerce(&Ty::unit(), &expected.ty);
            Ty::unit()
        };
        if diverges {
            Ty::simple(TypeCtor::Never)
        } else {
            ty
        }
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
        let resolved = method_resolution::lookup_method(
            &canonicalized_receiver.value,
            self.db,
            method_name,
            &self.resolver,
        );
        let (derefed_receiver_ty, method_ty, def_generics) = match resolved {
            Some((ty, func)) => {
                let ty = canonicalized_receiver.decanonicalize_ty(ty);
                self.write_method_resolution(tgt_expr, func);
                (ty, self.db.value_ty(func.into()), Some(generics(self.db, func.into())))
            }
            None => (receiver_ty, Ty::Unknown, None),
        };
        let substs = self.substs_for_method_call(def_generics, generic_args, &derefed_receiver_ty);
        let method_ty = method_ty.apply_substs(substs);
        let method_ty = self.insert_type_vars(method_ty);
        self.register_obligations_for_call(&method_ty);
        let (expected_receiver_ty, param_tys, ret_ty) = match method_ty.callable_sig(self.db) {
            Some(sig) => {
                if !sig.params().is_empty() {
                    (sig.params()[0].clone(), sig.params()[1..].to_vec(), sig.ret().clone())
                } else {
                    (Ty::Unknown, Vec::new(), sig.ret().clone())
                }
            }
            None => (Ty::Unknown, Vec::new(), Ty::Unknown),
        };
        // Apply autoref so the below unification works correctly
        // FIXME: return correct autorefs from lookup_method
        let actual_receiver_ty = match expected_receiver_ty.as_reference() {
            Some((_, mutability)) => Ty::apply_one(TypeCtor::Ref(mutability), derefed_receiver_ty),
            _ => derefed_receiver_ty,
        };
        self.unify(&expected_receiver_ty, &actual_receiver_ty);

        self.check_call_arguments(args, &param_tys);
        let ret_ty = self.normalize_associated_types_in(ret_ty);
        ret_ty
    }

    fn check_call_arguments(&mut self, args: &[ExprId], param_tys: &[Ty]) {
        // Quoting https://github.com/rust-lang/rust/blob/6ef275e6c3cb1384ec78128eceeb4963ff788dca/src/librustc_typeck/check/mod.rs#L3325 --
        // We do this in a pretty awful way: first we type-check any arguments
        // that are not closures, then we type-check the closures. This is so
        // that we have more information about the types of arguments when we
        // type-check the functions. This isn't really the right way to do this.
        for &check_closures in &[false, true] {
            let param_iter = param_tys.iter().cloned().chain(repeat(Ty::Unknown));
            for (&arg, param_ty) in args.iter().zip(param_iter) {
                let is_closure = match &self.body[arg] {
                    Expr::Lambda { .. } => true,
                    _ => false,
                };

                if is_closure != check_closures {
                    continue;
                }

                let param_ty = self.insert_vars_for_impl_trait(param_ty);
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
    ) -> Substs {
        let (total_len, _parent_len, child_len) =
            def_generics.as_ref().map_or((0, 0, 0), |g| g.len_split());
        let mut substs = Vec::with_capacity(total_len);
        // Parent arguments are unknown, except for the receiver type
        if let Some(parent_generics) = def_generics.as_ref().map(|p| p.iter_parent()) {
            for (_id, param) in parent_generics {
                if param.name == name![Self] {
                    substs.push(receiver_ty.clone());
                } else {
                    substs.push(Ty::Unknown);
                }
            }
        }
        // handle provided type arguments
        if let Some(generic_args) = generic_args {
            // if args are provided, it should be all of them, but we can't rely on that
            for arg in generic_args.args.iter().take(child_len) {
                match arg {
                    GenericArg::Type(type_ref) => {
                        let ty = self.make_ty(type_ref);
                        substs.push(ty);
                    }
                }
            }
        };
        let supplied_params = substs.len();
        for _ in supplied_params..total_len {
            substs.push(Ty::Unknown);
        }
        assert_eq!(substs.len(), total_len);
        Substs(substs.into())
    }

    fn register_obligations_for_call(&mut self, callable_ty: &Ty) {
        if let Ty::Apply(a_ty) = callable_ty {
            if let TypeCtor::FnDef(def) = a_ty.ctor {
                let generic_predicates = self.db.generic_predicates(def.into());
                for predicate in generic_predicates.iter() {
                    let predicate = predicate.clone().subst(&a_ty.parameters);
                    if let Some(obligation) = Obligation::from_predicate(predicate) {
                        self.obligations.push(obligation);
                    }
                }
                // add obligation for trait implementation, if this is a trait method
                match def {
                    CallableDef::FunctionId(f) => {
                        if let AssocContainerId::TraitId(trait_) = f.lookup(self.db).container {
                            // construct a TraitDef
                            let substs =
                                a_ty.parameters.prefix(generics(self.db, trait_.into()).len());
                            self.obligations.push(Obligation::Trait(TraitRef {
                                trait_: trait_.into(),
                                substs,
                            }));
                        }
                    }
                    CallableDef::StructId(_) | CallableDef::EnumVariantId(_) => {}
                }
            }
        }
    }
}
