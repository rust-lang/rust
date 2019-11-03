//! Type inference for expressions.

use std::iter::{repeat, repeat_with};
use std::sync::Arc;

use hir_def::path::{GenericArg, GenericArgs};
use hir_expand::name;

use super::{BindingMode, Expectation, InferenceContext, InferenceDiagnostic, TypeMismatch};
use crate::{
    db::HirDatabase,
    expr::{self, Array, BinaryOp, Expr, ExprId, Literal, Statement, UnaryOp},
    generics::{GenericParams, HasGenericParams},
    ty::{
        autoderef, method_resolution, op, primitive, CallableDef, InferTy, Mutability, Obligation,
        ProjectionPredicate, ProjectionTy, Substs, TraitRef, Ty, TypeCtor, TypeWalk,
    },
    Adt, Name, Namespace,
};

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
        let ty = self.resolve_ty_as_possible(&mut vec![], ty);
        ty
    }

    /// Infer type of expression with possibly implicit coerce to the expected type.
    /// Return the type after possible coercion.
    fn infer_expr_coerce(&mut self, expr: ExprId, expected: &Expectation) -> Ty {
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

        self.resolve_ty_as_possible(&mut vec![], ty)
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

                let pat_ty = match self.resolve_into_iter_item() {
                    Some(into_iter_item_alias) => {
                        let pat_ty = self.new_type_var();
                        let projection = ProjectionPredicate {
                            ty: pat_ty.clone(),
                            projection_ty: ProjectionTy {
                                associated_ty: into_iter_item_alias,
                                parameters: Substs::single(iterable_ty),
                            },
                        };
                        self.obligations.push(Obligation::Projection(projection));
                        self.resolve_ty_as_possible(&mut vec![], pat_ty)
                    }
                    None => Ty::Unknown,
                };

                self.infer_pat(*pat, &pat_ty, BindingMode::default());
                self.infer_expr(*body, &Expectation::has_type(Ty::unit()));
                Ty::unit()
            }
            Expr::Lambda { body, args, arg_types } => {
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
                let ret_ty = self.new_type_var();
                sig_tys.push(ret_ty.clone());
                let sig_ty = Ty::apply(
                    TypeCtor::FnPtr { num_args: sig_tys.len() as u16 - 1 },
                    Substs(sig_tys.into()),
                );
                let closure_ty = Ty::apply_one(
                    TypeCtor::Closure { def: self.body.owner(), expr: tgt_expr },
                    sig_ty,
                );

                // Eagerly try to relate the closure type with the expected
                // type, otherwise we often won't have enough information to
                // infer the body.
                self.coerce(&closure_ty, &expected.ty);

                self.infer_expr(*body, &Expectation::has_type(ret_ty));
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

                let mut result_ty = self.new_maybe_never_type_var();

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
                let resolver = expr::resolver_for_expr(self.body.clone(), self.db, tgt_expr);
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
                    self.infer_expr(*expr, &Expectation::has_type(self.return_ty.clone()));
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
                for (field_idx, field) in fields.iter().enumerate() {
                    let field_ty = def_id
                        .and_then(|it| match it.field(self.db, &field.name) {
                            Some(field) => Some(field),
                            None => {
                                self.push_diagnostic(InferenceDiagnostic::NoSuchField {
                                    expr: tgt_expr,
                                    field: field_idx,
                                });
                                None
                            }
                        })
                        .map_or(Ty::Unknown, |field| field.ty(self.db))
                        .subst(&substs);
                    self.infer_expr_coerce(field.expr, &Expectation::has_type(field_ty));
                }
                if let Some(expr) = spread {
                    self.infer_expr(*expr, &Expectation::has_type(ty.clone()));
                }
                ty
            }
            Expr::Field { expr, name } => {
                let receiver_ty = self.infer_expr(*expr, &Expectation::none());
                let canonicalized = self.canonicalizer().canonicalize_ty(receiver_ty);
                let ty = autoderef::autoderef(
                    self.db,
                    &self.resolver.clone(),
                    canonicalized.value.clone(),
                )
                .find_map(|derefed_ty| match canonicalized.decanonicalize_ty(derefed_ty.value) {
                    Ty::Apply(a_ty) => match a_ty.ctor {
                        TypeCtor::Tuple { .. } => name
                            .as_tuple_index()
                            .and_then(|idx| a_ty.parameters.0.get(idx).cloned()),
                        TypeCtor::Adt(Adt::Struct(s)) => s.field(self.db, name).map(|field| {
                            self.write_field_resolution(tgt_expr, field);
                            field.ty(self.db).subst(&a_ty.parameters)
                        }),
                        _ => None,
                    },
                    _ => None,
                })
                .unwrap_or(Ty::Unknown);
                let ty = self.insert_type_vars(ty);
                self.normalize_associated_types_in(ty)
            }
            Expr::Await { expr } => {
                let inner_ty = self.infer_expr(*expr, &Expectation::none());
                let ty = match self.resolve_future_future_output() {
                    Some(future_future_output_alias) => {
                        let ty = self.new_type_var();
                        let projection = ProjectionPredicate {
                            ty: ty.clone(),
                            projection_ty: ProjectionTy {
                                associated_ty: future_future_output_alias,
                                parameters: Substs::single(inner_ty),
                            },
                        };
                        self.obligations.push(Obligation::Projection(projection));
                        self.resolve_ty_as_possible(&mut vec![], ty)
                    }
                    None => Ty::Unknown,
                };
                ty
            }
            Expr::Try { expr } => {
                let inner_ty = self.infer_expr(*expr, &Expectation::none());
                let ty = match self.resolve_ops_try_ok() {
                    Some(ops_try_ok_alias) => {
                        let ty = self.new_type_var();
                        let projection = ProjectionPredicate {
                            ty: ty.clone(),
                            projection_ty: ProjectionTy {
                                associated_ty: ops_try_ok_alias,
                                parameters: Substs::single(inner_ty),
                            },
                        };
                        self.obligations.push(Obligation::Projection(projection));
                        self.resolve_ty_as_possible(&mut vec![], ty)
                    }
                    None => Ty::Unknown,
                };
                ty
            }
            Expr::Cast { expr, type_ref } => {
                let _inner_ty = self.infer_expr(*expr, &Expectation::none());
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
                // FIXME reference coercions etc.
                let inner_ty = self.infer_expr(*expr, &expectation);
                Ty::apply_one(TypeCtor::Ref(*mutability), inner_ty)
            }
            Expr::Box { expr } => {
                let inner_ty = self.infer_expr(*expr, &Expectation::none());
                if let Some(box_) = self.resolve_boxed_box() {
                    Ty::apply_one(TypeCtor::Adt(box_), inner_ty)
                } else {
                    Ty::Unknown
                }
            }
            Expr::UnaryOp { expr, op } => {
                let inner_ty = self.infer_expr(*expr, &Expectation::none());
                match op {
                    UnaryOp::Deref => {
                        let canonicalized = self.canonicalizer().canonicalize_ty(inner_ty);
                        if let Some(derefed_ty) =
                            autoderef::deref(self.db, &self.resolver, &canonicalized.value)
                        {
                            canonicalized.decanonicalize_ty(derefed_ty.value)
                        } else {
                            Ty::Unknown
                        }
                    }
                    UnaryOp::Neg => {
                        match &inner_ty {
                            Ty::Apply(a_ty) => match a_ty.ctor {
                                TypeCtor::Int(primitive::UncertainIntTy::Unknown)
                                | TypeCtor::Int(primitive::UncertainIntTy::Known(
                                    primitive::IntTy {
                                        signedness: primitive::Signedness::Signed,
                                        ..
                                    },
                                ))
                                | TypeCtor::Float(..) => inner_ty,
                                _ => Ty::Unknown,
                            },
                            Ty::Infer(InferTy::IntVar(..)) | Ty::Infer(InferTy::FloatVar(..)) => {
                                inner_ty
                            }
                            // FIXME: resolve ops::Neg trait
                            _ => Ty::Unknown,
                        }
                    }
                    UnaryOp::Not => {
                        match &inner_ty {
                            Ty::Apply(a_ty) => match a_ty.ctor {
                                TypeCtor::Bool | TypeCtor::Int(_) => inner_ty,
                                _ => Ty::Unknown,
                            },
                            Ty::Infer(InferTy::IntVar(..)) => inner_ty,
                            // FIXME: resolve ops::Not trait for inner_ty
                            _ => Ty::Unknown,
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
            Expr::Index { base, index } => {
                let _base_ty = self.infer_expr(*base, &Expectation::none());
                let _index_ty = self.infer_expr(*index, &Expectation::none());
                // FIXME: use `std::ops::Index::Output` to figure out the real return type
                Ty::Unknown
            }
            Expr::Tuple { exprs } => {
                let mut tys = match &expected.ty {
                    ty_app!(TypeCtor::Tuple { .. }, st) => st
                        .iter()
                        .cloned()
                        .chain(repeat_with(|| self.new_type_var()))
                        .take(exprs.len())
                        .collect::<Vec<_>>(),
                    _ => (0..exprs.len()).map(|_| self.new_type_var()).collect(),
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
                    _ => self.new_type_var(),
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
                            &Expectation::has_type(Ty::simple(TypeCtor::Int(
                                primitive::UncertainIntTy::Known(primitive::IntTy::usize()),
                            ))),
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
                    let byte_type = Ty::simple(TypeCtor::Int(primitive::UncertainIntTy::Known(
                        primitive::IntTy::u8(),
                    )));
                    let slice_type = Ty::apply_one(TypeCtor::Slice, byte_type);
                    Ty::apply_one(TypeCtor::Ref(Mutability::Shared), slice_type)
                }
                Literal::Char(..) => Ty::simple(TypeCtor::Char),
                Literal::Int(_v, ty) => Ty::simple(TypeCtor::Int(*ty)),
                Literal::Float(_v, ty) => Ty::simple(TypeCtor::Float(*ty)),
            },
        };
        // use a new type variable if we got Ty::Unknown here
        let ty = self.insert_type_vars_shallow(ty);
        let ty = self.resolve_ty_as_possible(&mut vec![], ty);
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

                    let ty = self.resolve_ty_as_possible(&mut vec![], ty);
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
                (
                    ty,
                    self.db.type_for_def(func.into(), Namespace::Values),
                    Some(func.generic_params(self.db)),
                )
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

                let param_ty = self.normalize_associated_types_in(param_ty);
                self.infer_expr_coerce(arg, &Expectation::has_type(param_ty.clone()));
            }
        }
    }

    fn substs_for_method_call(
        &mut self,
        def_generics: Option<Arc<GenericParams>>,
        generic_args: Option<&GenericArgs>,
        receiver_ty: &Ty,
    ) -> Substs {
        let (parent_param_count, param_count) =
            def_generics.as_ref().map_or((0, 0), |g| (g.count_parent_params(), g.params.len()));
        let mut substs = Vec::with_capacity(parent_param_count + param_count);
        // Parent arguments are unknown, except for the receiver type
        if let Some(parent_generics) = def_generics.and_then(|p| p.parent_params.clone()) {
            for param in &parent_generics.params {
                if param.name == name::SELF_TYPE {
                    substs.push(receiver_ty.clone());
                } else {
                    substs.push(Ty::Unknown);
                }
            }
        }
        // handle provided type arguments
        if let Some(generic_args) = generic_args {
            // if args are provided, it should be all of them, but we can't rely on that
            for arg in generic_args.args.iter().take(param_count) {
                match arg {
                    GenericArg::Type(type_ref) => {
                        let ty = self.make_ty(type_ref);
                        substs.push(ty);
                    }
                }
            }
        };
        let supplied_params = substs.len();
        for _ in supplied_params..parent_param_count + param_count {
            substs.push(Ty::Unknown);
        }
        assert_eq!(substs.len(), parent_param_count + param_count);
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
                    CallableDef::Function(f) => {
                        if let Some(trait_) = f.parent_trait(self.db) {
                            // construct a TraitDef
                            let substs = a_ty.parameters.prefix(
                                trait_.generic_params(self.db).count_params_including_parent(),
                            );
                            self.obligations.push(Obligation::Trait(TraitRef { trait_, substs }));
                        }
                    }
                    CallableDef::Struct(_) | CallableDef::EnumVariant(_) => {}
                }
            }
        }
    }
}
