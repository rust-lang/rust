//! Inference of closure parameter types based on the closure's expected type.

use std::{cmp, convert::Infallible, mem, ops::ControlFlow};

use chalk_ir::{
    BoundVar, DebruijnIndex, FnSubst, Mutability, TyKind,
    cast::Cast,
    fold::{FallibleTypeFolder, Shift, TypeFoldable},
    visit::{TypeSuperVisitable, TypeVisitable, TypeVisitor},
};
use either::Either;
use hir_def::{
    DefWithBodyId, FieldId, HasModule, TupleFieldId, TupleId, VariantId,
    expr_store::path::Path,
    hir::{
        Array, AsmOperand, BinaryOp, BindingId, CaptureBy, ClosureKind, Expr, ExprId, ExprOrPatId,
        Pat, PatId, Statement, UnaryOp,
    },
    item_tree::FieldsShape,
    lang_item::LangItem,
    resolver::ValueNs,
};
use hir_def::{Lookup, type_ref::TypeRefId};
use hir_expand::name::Name;
use intern::sym;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{SmallVec, smallvec};
use stdx::{format_to, never};
use syntax::utils::is_raw_identifier;

use crate::{
    Adjust, Adjustment, AliasEq, AliasTy, Binders, BindingMode, ChalkTraitId, ClosureId, DynTy,
    DynTyExt, FnAbi, FnPointer, FnSig, GenericArg, Interner, OpaqueTy, ProjectionTy,
    ProjectionTyExt, Substitution, Ty, TyBuilder, TyExt, WhereClause,
    db::{HirDatabase, InternedClosure, InternedCoroutine},
    error_lifetime, from_assoc_type_id, from_chalk_trait_id, from_placeholder_idx,
    generics::Generics,
    infer::{BreakableKind, CoerceMany, Diverges, coerce::CoerceNever},
    make_binders,
    mir::{BorrowKind, MirSpan, MutBorrowKind, ProjectionElem},
    to_assoc_type_id, to_chalk_trait_id,
    traits::FnTrait,
    utils::{self, elaborate_clause_supertraits},
};

use super::{Expectation, InferenceContext};

#[derive(Debug)]
pub(super) struct ClosureSignature {
    pub(super) ret_ty: Ty,
    pub(super) expected_sig: FnPointer,
}

impl InferenceContext<'_> {
    pub(super) fn infer_closure(
        &mut self,
        body: &ExprId,
        args: &[PatId],
        ret_type: &Option<TypeRefId>,
        arg_types: &[Option<TypeRefId>],
        closure_kind: ClosureKind,
        tgt_expr: ExprId,
        expected: &Expectation,
    ) -> Ty {
        assert_eq!(args.len(), arg_types.len());

        let (expected_sig, expected_kind) = match expected.to_option(&mut self.table) {
            Some(expected_ty) => self.deduce_closure_signature(&expected_ty, closure_kind),
            None => (None, None),
        };

        let ClosureSignature { expected_sig: bound_sig, ret_ty: body_ret_ty } =
            self.sig_of_closure(body, ret_type, arg_types, closure_kind, expected_sig);
        let bound_sig = self.normalize_associated_types_in(bound_sig);
        let sig_ty = TyKind::Function(bound_sig.clone()).intern(Interner);

        let (id, ty, resume_yield_tys) = match closure_kind {
            ClosureKind::Coroutine(_) => {
                let sig_tys = bound_sig.substitution.0.as_slice(Interner);
                // FIXME: report error when there are more than 1 parameter.
                let resume_ty = match sig_tys.first() {
                    // When `sig_tys.len() == 1` the first type is the return type, not the
                    // first parameter type.
                    Some(ty) if sig_tys.len() > 1 => ty.assert_ty_ref(Interner).clone(),
                    _ => self.result.standard_types.unit.clone(),
                };
                let yield_ty = self.table.new_type_var();

                let subst = TyBuilder::subst_for_coroutine(self.db, self.owner)
                    .push(resume_ty.clone())
                    .push(yield_ty.clone())
                    .push(body_ret_ty.clone())
                    .build();

                let coroutine_id =
                    self.db.intern_coroutine(InternedCoroutine(self.owner, tgt_expr)).into();
                let coroutine_ty = TyKind::Coroutine(coroutine_id, subst).intern(Interner);

                (None, coroutine_ty, Some((resume_ty, yield_ty)))
            }
            ClosureKind::Closure | ClosureKind::Async => {
                let closure_id =
                    self.db.intern_closure(InternedClosure(self.owner, tgt_expr)).into();
                let closure_ty = TyKind::Closure(
                    closure_id,
                    TyBuilder::subst_for_closure(self.db, self.owner, sig_ty.clone()),
                )
                .intern(Interner);
                self.deferred_closures.entry(closure_id).or_default();
                self.add_current_closure_dependency(closure_id);
                (Some(closure_id), closure_ty, None)
            }
        };

        // Eagerly try to relate the closure type with the expected
        // type, otherwise we often won't have enough information to
        // infer the body.
        self.deduce_closure_type_from_expectations(tgt_expr, &ty, &sig_ty, expected, expected_kind);

        // Now go through the argument patterns
        for (arg_pat, arg_ty) in args.iter().zip(bound_sig.substitution.0.as_slice(Interner).iter())
        {
            self.infer_top_pat(*arg_pat, arg_ty.assert_ty_ref(Interner), None);
        }

        // FIXME: lift these out into a struct
        let prev_diverges = mem::replace(&mut self.diverges, Diverges::Maybe);
        let prev_closure = mem::replace(&mut self.current_closure, id);
        let prev_ret_ty = mem::replace(&mut self.return_ty, body_ret_ty.clone());
        let prev_ret_coercion = self.return_coercion.replace(CoerceMany::new(body_ret_ty));
        let prev_resume_yield_tys = mem::replace(&mut self.resume_yield_tys, resume_yield_tys);

        self.with_breakable_ctx(BreakableKind::Border, None, None, |this| {
            this.infer_return(*body);
        });

        self.diverges = prev_diverges;
        self.return_ty = prev_ret_ty;
        self.return_coercion = prev_ret_coercion;
        self.current_closure = prev_closure;
        self.resume_yield_tys = prev_resume_yield_tys;

        self.table.normalize_associated_types_in(ty)
    }

    // This function handles both closures and coroutines.
    pub(super) fn deduce_closure_type_from_expectations(
        &mut self,
        closure_expr: ExprId,
        closure_ty: &Ty,
        sig_ty: &Ty,
        expectation: &Expectation,
        expected_kind: Option<FnTrait>,
    ) {
        let expected_ty = match expectation.to_option(&mut self.table) {
            Some(ty) => ty,
            None => return,
        };

        match (closure_ty.kind(Interner), expected_kind) {
            (TyKind::Closure(closure_id, _), Some(closure_kind)) => {
                self.result
                    .closure_info
                    .entry(*closure_id)
                    .or_insert_with(|| (Vec::new(), closure_kind));
            }
            _ => {}
        }

        // Deduction from where-clauses in scope, as well as fn-pointer coercion are handled here.
        let _ = self.coerce(Some(closure_expr), closure_ty, &expected_ty, CoerceNever::Yes);

        // Coroutines are not Fn* so return early.
        if matches!(closure_ty.kind(Interner), TyKind::Coroutine(..)) {
            return;
        }

        // Deduction based on the expected `dyn Fn` is done separately.
        if let TyKind::Dyn(dyn_ty) = expected_ty.kind(Interner) {
            if let Some(sig) = self.deduce_sig_from_dyn_ty(dyn_ty) {
                let expected_sig_ty = TyKind::Function(sig).intern(Interner);

                self.unify(sig_ty, &expected_sig_ty);
            }
        }
    }

    // Closure kind deductions are mostly from `rustc_hir_typeck/src/closure.rs`.
    // Might need to port closure sig deductions too.
    pub(super) fn deduce_closure_signature(
        &mut self,
        expected_ty: &Ty,
        closure_kind: ClosureKind,
    ) -> (Option<FnSubst<Interner>>, Option<FnTrait>) {
        match expected_ty.kind(Interner) {
            TyKind::Alias(AliasTy::Opaque(OpaqueTy { .. })) | TyKind::OpaqueType(..) => {
                let clauses = expected_ty.impl_trait_bounds(self.db).into_iter().flatten().map(
                    |b: chalk_ir::Binders<chalk_ir::WhereClause<Interner>>| {
                        b.into_value_and_skipped_binders().0
                    },
                );
                self.deduce_closure_kind_from_predicate_clauses(expected_ty, clauses, closure_kind)
            }
            TyKind::Dyn(dyn_ty) => {
                let sig =
                    dyn_ty.bounds.skip_binders().as_slice(Interner).iter().find_map(|bound| {
                        if let WhereClause::AliasEq(AliasEq {
                            alias: AliasTy::Projection(projection_ty),
                            ty: projected_ty,
                        }) = bound.skip_binders()
                        {
                            if let Some(sig) = self.deduce_sig_from_projection(
                                closure_kind,
                                projection_ty,
                                projected_ty,
                            ) {
                                return Some(sig);
                            }
                        }
                        None
                    });

                let kind = dyn_ty.principal().and_then(|principal_trait_ref| {
                    self.fn_trait_kind_from_trait_id(from_chalk_trait_id(
                        principal_trait_ref.skip_binders().skip_binders().trait_id,
                    ))
                });

                (sig, kind)
            }
            TyKind::InferenceVar(ty, chalk_ir::TyVariableKind::General) => {
                let clauses = self.clauses_for_self_ty(*ty);
                self.deduce_closure_kind_from_predicate_clauses(
                    expected_ty,
                    clauses.into_iter(),
                    closure_kind,
                )
            }
            TyKind::Function(fn_ptr) => match closure_kind {
                ClosureKind::Closure => (Some(fn_ptr.substitution.clone()), Some(FnTrait::Fn)),
                ClosureKind::Async | ClosureKind::Coroutine(_) => (None, None),
            },
            _ => (None, None),
        }
    }

    fn deduce_closure_kind_from_predicate_clauses(
        &mut self,
        expected_ty: &Ty,
        clauses: impl DoubleEndedIterator<Item = WhereClause>,
        closure_kind: ClosureKind,
    ) -> (Option<FnSubst<Interner>>, Option<FnTrait>) {
        let mut expected_sig = None;
        let mut expected_kind = None;

        for clause in elaborate_clause_supertraits(self.db, clauses.rev()) {
            if expected_sig.is_none() {
                if let WhereClause::AliasEq(AliasEq {
                    alias: AliasTy::Projection(projection),
                    ty,
                }) = &clause
                {
                    let inferred_sig =
                        self.deduce_sig_from_projection(closure_kind, projection, ty);
                    // Make sure that we didn't infer a signature that mentions itself.
                    // This can happen when we elaborate certain supertrait bounds that
                    // mention projections containing the `Self` type. See rust-lang/rust#105401.
                    struct MentionsTy<'a> {
                        expected_ty: &'a Ty,
                    }
                    impl TypeVisitor<Interner> for MentionsTy<'_> {
                        type BreakTy = ();

                        fn interner(&self) -> Interner {
                            Interner
                        }

                        fn as_dyn(
                            &mut self,
                        ) -> &mut dyn TypeVisitor<Interner, BreakTy = Self::BreakTy>
                        {
                            self
                        }

                        fn visit_ty(
                            &mut self,
                            t: &Ty,
                            db: chalk_ir::DebruijnIndex,
                        ) -> ControlFlow<()> {
                            if t == self.expected_ty {
                                ControlFlow::Break(())
                            } else {
                                t.super_visit_with(self, db)
                            }
                        }
                    }
                    if inferred_sig
                        .visit_with(
                            &mut MentionsTy { expected_ty },
                            chalk_ir::DebruijnIndex::INNERMOST,
                        )
                        .is_continue()
                    {
                        expected_sig = inferred_sig;
                    }
                }
            }

            let trait_id = match clause {
                WhereClause::AliasEq(AliasEq {
                    alias: AliasTy::Projection(projection), ..
                }) => projection.trait_(self.db),
                WhereClause::Implemented(trait_ref) => from_chalk_trait_id(trait_ref.trait_id),
                _ => continue,
            };
            if let Some(closure_kind) = self.fn_trait_kind_from_trait_id(trait_id) {
                // always use the closure kind that is more permissive.
                match (expected_kind, closure_kind) {
                    (None, _) => expected_kind = Some(closure_kind),
                    (Some(FnTrait::FnMut), FnTrait::Fn) => expected_kind = Some(FnTrait::Fn),
                    (Some(FnTrait::FnOnce), FnTrait::Fn | FnTrait::FnMut) => {
                        expected_kind = Some(closure_kind)
                    }
                    _ => {}
                }
            }
        }

        (expected_sig, expected_kind)
    }

    fn deduce_sig_from_dyn_ty(&self, dyn_ty: &DynTy) -> Option<FnPointer> {
        // Search for a predicate like `<$self as FnX<Args>>::Output == Ret`

        let fn_traits: SmallVec<[ChalkTraitId; 3]> =
            utils::fn_traits(self.db, self.owner.module(self.db).krate())
                .map(to_chalk_trait_id)
                .collect();

        let self_ty = self.result.standard_types.unknown.clone();
        let bounds = dyn_ty.bounds.clone().substitute(Interner, &[self_ty.cast(Interner)]);
        for bound in bounds.iter(Interner) {
            // NOTE(skip_binders): the extracted types are rebound by the returned `FnPointer`
            if let WhereClause::AliasEq(AliasEq { alias: AliasTy::Projection(projection), ty }) =
                bound.skip_binders()
            {
                let assoc_data =
                    self.db.associated_ty_data(from_assoc_type_id(projection.associated_ty_id));
                if !fn_traits.contains(&assoc_data.trait_id) {
                    return None;
                }

                // Skip `Self`, get the type argument.
                let arg = projection.substitution.as_slice(Interner).get(1)?;
                if let Some(subst) = arg.ty(Interner)?.as_tuple() {
                    let generic_args = subst.as_slice(Interner);
                    let mut sig_tys = Vec::with_capacity(generic_args.len() + 1);
                    for arg in generic_args {
                        sig_tys.push(arg.ty(Interner)?.clone());
                    }
                    sig_tys.push(ty.clone());

                    cov_mark::hit!(dyn_fn_param_informs_call_site_closure_signature);
                    return Some(FnPointer {
                        num_binders: bound.len(Interner),
                        sig: FnSig {
                            abi: FnAbi::RustCall,
                            safety: chalk_ir::Safety::Safe,
                            variadic: false,
                        },
                        substitution: FnSubst(Substitution::from_iter(Interner, sig_tys)),
                    });
                }
            }
        }

        None
    }

    fn deduce_sig_from_projection(
        &mut self,
        closure_kind: ClosureKind,
        projection_ty: &ProjectionTy,
        projected_ty: &Ty,
    ) -> Option<FnSubst<Interner>> {
        let container =
            from_assoc_type_id(projection_ty.associated_ty_id).lookup(self.db).container;
        let trait_ = match container {
            hir_def::ItemContainerId::TraitId(trait_) => trait_,
            _ => return None,
        };

        // For now, we only do signature deduction based off of the `Fn` and `AsyncFn` traits,
        // for closures and async closures, respectively.
        let fn_trait_kind = self.fn_trait_kind_from_trait_id(trait_)?;
        if !matches!(closure_kind, ClosureKind::Closure | ClosureKind::Async) {
            return None;
        }
        if fn_trait_kind.is_async() {
            // If the expected trait is `AsyncFn(...) -> X`, we don't know what the return type is,
            // but we do know it must implement `Future<Output = X>`.
            self.extract_async_fn_sig_from_projection(projection_ty, projected_ty)
        } else {
            self.extract_sig_from_projection(projection_ty, projected_ty)
        }
    }

    fn extract_sig_from_projection(
        &self,
        projection_ty: &ProjectionTy,
        projected_ty: &Ty,
    ) -> Option<FnSubst<Interner>> {
        let arg_param_ty = projection_ty.substitution.as_slice(Interner)[1].assert_ty_ref(Interner);

        let TyKind::Tuple(_, input_tys) = arg_param_ty.kind(Interner) else {
            return None;
        };

        let ret_param_ty = projected_ty;

        Some(FnSubst(Substitution::from_iter(
            Interner,
            input_tys.iter(Interner).map(|t| t.cast(Interner)).chain(Some(GenericArg::new(
                Interner,
                chalk_ir::GenericArgData::Ty(ret_param_ty.clone()),
            ))),
        )))
    }

    fn extract_async_fn_sig_from_projection(
        &mut self,
        projection_ty: &ProjectionTy,
        projected_ty: &Ty,
    ) -> Option<FnSubst<Interner>> {
        let arg_param_ty = projection_ty.substitution.as_slice(Interner)[1].assert_ty_ref(Interner);

        let TyKind::Tuple(_, input_tys) = arg_param_ty.kind(Interner) else {
            return None;
        };

        let ret_param_future_output = projected_ty;
        let ret_param_future = self.table.new_type_var();
        let future_output =
            LangItem::FutureOutput.resolve_type_alias(self.db, self.resolver.krate())?;
        let future_projection = crate::AliasTy::Projection(crate::ProjectionTy {
            associated_ty_id: to_assoc_type_id(future_output),
            substitution: Substitution::from1(Interner, ret_param_future.clone()),
        });
        self.table.register_obligation(
            crate::AliasEq { alias: future_projection, ty: ret_param_future_output.clone() }
                .cast(Interner),
        );

        Some(FnSubst(Substitution::from_iter(
            Interner,
            input_tys.iter(Interner).map(|t| t.cast(Interner)).chain(Some(GenericArg::new(
                Interner,
                chalk_ir::GenericArgData::Ty(ret_param_future),
            ))),
        )))
    }

    fn fn_trait_kind_from_trait_id(&self, trait_id: hir_def::TraitId) -> Option<FnTrait> {
        FnTrait::from_lang_item(self.db.lang_attr(trait_id.into())?)
    }

    fn supplied_sig_of_closure(
        &mut self,
        body: &ExprId,
        ret_type: &Option<TypeRefId>,
        arg_types: &[Option<TypeRefId>],
        closure_kind: ClosureKind,
    ) -> ClosureSignature {
        let mut sig_tys = Vec::with_capacity(arg_types.len() + 1);

        // collect explicitly written argument types
        for arg_type in arg_types.iter() {
            let arg_ty = match arg_type {
                // FIXME: I think rustc actually lowers closure params with `LifetimeElisionKind::AnonymousCreateParameter`
                // (but the return type with infer).
                Some(type_ref) => self.make_body_ty(*type_ref),
                None => self.table.new_type_var(),
            };
            sig_tys.push(arg_ty);
        }

        // add return type
        let ret_ty = match ret_type {
            Some(type_ref) => self.make_body_ty(*type_ref),
            None => self.table.new_type_var(),
        };
        if let ClosureKind::Async = closure_kind {
            sig_tys.push(self.lower_async_block_type_impl_trait(ret_ty.clone(), *body));
        } else {
            sig_tys.push(ret_ty.clone());
        }

        let expected_sig = FnPointer {
            num_binders: 0,
            sig: FnSig { abi: FnAbi::RustCall, safety: chalk_ir::Safety::Safe, variadic: false },
            substitution: FnSubst(
                Substitution::from_iter(Interner, sig_tys.iter().cloned()).shifted_in(Interner),
            ),
        };

        ClosureSignature { ret_ty, expected_sig }
    }

    /// The return type is the signature of the closure, and the return type
    /// *as represented inside the body* (so, for async closures, the `Output` ty)
    pub(super) fn sig_of_closure(
        &mut self,
        body: &ExprId,
        ret_type: &Option<TypeRefId>,
        arg_types: &[Option<TypeRefId>],
        closure_kind: ClosureKind,
        expected_sig: Option<FnSubst<Interner>>,
    ) -> ClosureSignature {
        if let Some(e) = expected_sig {
            self.sig_of_closure_with_expectation(body, ret_type, arg_types, closure_kind, e)
        } else {
            self.sig_of_closure_no_expectation(body, ret_type, arg_types, closure_kind)
        }
    }

    fn sig_of_closure_no_expectation(
        &mut self,
        body: &ExprId,
        ret_type: &Option<TypeRefId>,
        arg_types: &[Option<TypeRefId>],
        closure_kind: ClosureKind,
    ) -> ClosureSignature {
        self.supplied_sig_of_closure(body, ret_type, arg_types, closure_kind)
    }

    fn sig_of_closure_with_expectation(
        &mut self,
        body: &ExprId,
        ret_type: &Option<TypeRefId>,
        arg_types: &[Option<TypeRefId>],
        closure_kind: ClosureKind,
        expected_sig: FnSubst<Interner>,
    ) -> ClosureSignature {
        let expected_sig = FnPointer {
            num_binders: 0,
            sig: FnSig { abi: FnAbi::RustCall, safety: chalk_ir::Safety::Safe, variadic: false },
            substitution: expected_sig,
        };

        // If the expected signature does not match the actual arg types,
        // then just return the expected signature
        if expected_sig.substitution.0.len(Interner) != arg_types.len() + 1 {
            let ret_ty = match ret_type {
                Some(type_ref) => self.make_body_ty(*type_ref),
                None => self.table.new_type_var(),
            };
            return ClosureSignature { expected_sig, ret_ty };
        }

        self.merge_supplied_sig_with_expectation(
            body,
            ret_type,
            arg_types,
            closure_kind,
            expected_sig,
        )
    }

    fn merge_supplied_sig_with_expectation(
        &mut self,
        body: &ExprId,
        ret_type: &Option<TypeRefId>,
        arg_types: &[Option<TypeRefId>],
        closure_kind: ClosureKind,
        expected_sig: FnPointer,
    ) -> ClosureSignature {
        let supplied_sig = self.supplied_sig_of_closure(body, ret_type, arg_types, closure_kind);

        let snapshot = self.table.snapshot();
        if !self.table.unify(&expected_sig.substitution, &supplied_sig.expected_sig.substitution) {
            self.table.rollback_to(snapshot);
        }

        supplied_sig
    }
}

// The below functions handle capture and closure kind (Fn, FnMut, ..)

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct HirPlace {
    pub(crate) local: BindingId,
    pub(crate) projections: Vec<ProjectionElem<Infallible, Ty>>,
}

impl HirPlace {
    fn ty(&self, ctx: &mut InferenceContext<'_>) -> Ty {
        let mut ty = ctx.table.resolve_completely(ctx.result[self.local].clone());
        for p in &self.projections {
            ty = p.projected_ty(
                ty,
                ctx.db,
                |_, _, _| {
                    unreachable!("Closure field only happens in MIR");
                },
                ctx.owner.module(ctx.db).krate(),
            );
        }
        ty
    }

    fn capture_kind_of_truncated_place(
        &self,
        mut current_capture: CaptureKind,
        len: usize,
    ) -> CaptureKind {
        if let CaptureKind::ByRef(BorrowKind::Mut {
            kind: MutBorrowKind::Default | MutBorrowKind::TwoPhasedBorrow,
        }) = current_capture
        {
            if self.projections[len..].contains(&ProjectionElem::Deref) {
                current_capture =
                    CaptureKind::ByRef(BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture });
            }
        }
        current_capture
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CaptureKind {
    ByRef(BorrowKind),
    ByValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedItem {
    pub(crate) place: HirPlace,
    pub(crate) kind: CaptureKind,
    /// The inner vec is the stacks; the outer vec is for each capture reference.
    ///
    /// Even though we always report only the last span (i.e. the most inclusive span),
    /// we need to keep them all, since when a closure occurs inside a closure, we
    /// copy all captures of the inner closure to the outer closure, and then we may
    /// truncate them, and we want the correct span to be reported.
    span_stacks: SmallVec<[SmallVec<[MirSpan; 3]>; 3]>,
    pub(crate) ty: Binders<Ty>,
}

impl CapturedItem {
    pub fn local(&self) -> BindingId {
        self.place.local
    }

    /// Returns whether this place has any field (aka. non-deref) projections.
    pub fn has_field_projections(&self) -> bool {
        self.place.projections.iter().any(|it| !matches!(it, ProjectionElem::Deref))
    }

    pub fn ty(&self, subst: &Substitution) -> Ty {
        self.ty.clone().substitute(Interner, utils::ClosureSubst(subst).parent_subst())
    }

    pub fn kind(&self) -> CaptureKind {
        self.kind
    }

    pub fn spans(&self) -> SmallVec<[MirSpan; 3]> {
        self.span_stacks.iter().map(|stack| *stack.last().expect("empty span stack")).collect()
    }

    /// Converts the place to a name that can be inserted into source code.
    pub fn place_to_name(&self, owner: DefWithBodyId, db: &dyn HirDatabase) -> String {
        let body = db.body(owner);
        let mut result = body[self.place.local].name.as_str().to_owned();
        for proj in &self.place.projections {
            match proj {
                ProjectionElem::Deref => {}
                ProjectionElem::Field(Either::Left(f)) => {
                    let variant_data = f.parent.fields(db);
                    match variant_data.shape {
                        FieldsShape::Record => {
                            result.push('_');
                            result.push_str(variant_data.fields()[f.local_id].name.as_str())
                        }
                        FieldsShape::Tuple => {
                            let index =
                                variant_data.fields().iter().position(|it| it.0 == f.local_id);
                            if let Some(index) = index {
                                format_to!(result, "_{index}");
                            }
                        }
                        FieldsShape::Unit => {}
                    }
                }
                ProjectionElem::Field(Either::Right(f)) => format_to!(result, "_{}", f.index),
                &ProjectionElem::ClosureField(field) => format_to!(result, "_{field}"),
                ProjectionElem::Index(_)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::OpaqueCast(_) => {
                    never!("Not happen in closure capture");
                    continue;
                }
            }
        }
        if is_raw_identifier(&result, owner.module(db).krate().data(db).edition) {
            result.insert_str(0, "r#");
        }
        result
    }

    pub fn display_place_source_code(&self, owner: DefWithBodyId, db: &dyn HirDatabase) -> String {
        let body = db.body(owner);
        let krate = owner.krate(db);
        let edition = krate.data(db).edition;
        let mut result = body[self.place.local].name.display(db, edition).to_string();
        for proj in &self.place.projections {
            match proj {
                // In source code autoderef kicks in.
                ProjectionElem::Deref => {}
                ProjectionElem::Field(Either::Left(f)) => {
                    let variant_data = f.parent.fields(db);
                    match variant_data.shape {
                        FieldsShape::Record => format_to!(
                            result,
                            ".{}",
                            variant_data.fields()[f.local_id].name.display(db, edition)
                        ),
                        FieldsShape::Tuple => format_to!(
                            result,
                            ".{}",
                            variant_data
                                .fields()
                                .iter()
                                .position(|it| it.0 == f.local_id)
                                .unwrap_or_default()
                        ),
                        FieldsShape::Unit => {}
                    }
                }
                ProjectionElem::Field(Either::Right(f)) => {
                    let field = f.index;
                    format_to!(result, ".{field}");
                }
                &ProjectionElem::ClosureField(field) => {
                    format_to!(result, ".{field}");
                }
                ProjectionElem::Index(_)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::OpaqueCast(_) => {
                    never!("Not happen in closure capture");
                    continue;
                }
            }
        }
        let final_derefs_count = self
            .place
            .projections
            .iter()
            .rev()
            .take_while(|proj| matches!(proj, ProjectionElem::Deref))
            .count();
        result.insert_str(0, &"*".repeat(final_derefs_count));
        result
    }

    pub fn display_place(&self, owner: DefWithBodyId, db: &dyn HirDatabase) -> String {
        let body = db.body(owner);
        let krate = owner.krate(db);
        let edition = krate.data(db).edition;
        let mut result = body[self.place.local].name.display(db, edition).to_string();
        let mut field_need_paren = false;
        for proj in &self.place.projections {
            match proj {
                ProjectionElem::Deref => {
                    result = format!("*{result}");
                    field_need_paren = true;
                }
                ProjectionElem::Field(Either::Left(f)) => {
                    if field_need_paren {
                        result = format!("({result})");
                    }
                    let variant_data = f.parent.fields(db);
                    let field = match variant_data.shape {
                        FieldsShape::Record => {
                            variant_data.fields()[f.local_id].name.as_str().to_owned()
                        }
                        FieldsShape::Tuple => variant_data
                            .fields()
                            .iter()
                            .position(|it| it.0 == f.local_id)
                            .unwrap_or_default()
                            .to_string(),
                        FieldsShape::Unit => "[missing field]".to_owned(),
                    };
                    result = format!("{result}.{field}");
                    field_need_paren = false;
                }
                ProjectionElem::Field(Either::Right(f)) => {
                    let field = f.index;
                    if field_need_paren {
                        result = format!("({result})");
                    }
                    result = format!("{result}.{field}");
                    field_need_paren = false;
                }
                &ProjectionElem::ClosureField(field) => {
                    if field_need_paren {
                        result = format!("({result})");
                    }
                    result = format!("{result}.{field}");
                    field_need_paren = false;
                }
                ProjectionElem::Index(_)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::OpaqueCast(_) => {
                    never!("Not happen in closure capture");
                    continue;
                }
            }
        }
        result
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CapturedItemWithoutTy {
    pub(crate) place: HirPlace,
    pub(crate) kind: CaptureKind,
    /// The inner vec is the stacks; the outer vec is for each capture reference.
    pub(crate) span_stacks: SmallVec<[SmallVec<[MirSpan; 3]>; 3]>,
}

impl CapturedItemWithoutTy {
    fn with_ty(self, ctx: &mut InferenceContext<'_>) -> CapturedItem {
        let ty = self.place.ty(ctx);
        let ty = match &self.kind {
            CaptureKind::ByValue => ty,
            CaptureKind::ByRef(bk) => {
                let m = match bk {
                    BorrowKind::Mut { .. } => Mutability::Mut,
                    _ => Mutability::Not,
                };
                TyKind::Ref(m, error_lifetime(), ty).intern(Interner)
            }
        };
        return CapturedItem {
            place: self.place,
            kind: self.kind,
            span_stacks: self.span_stacks,
            ty: replace_placeholder_with_binder(ctx, ty),
        };

        fn replace_placeholder_with_binder(ctx: &mut InferenceContext<'_>, ty: Ty) -> Binders<Ty> {
            struct Filler<'a> {
                db: &'a dyn HirDatabase,
                generics: &'a Generics,
            }
            impl FallibleTypeFolder<Interner> for Filler<'_> {
                type Error = ();

                fn as_dyn(&mut self) -> &mut dyn FallibleTypeFolder<Interner, Error = Self::Error> {
                    self
                }

                fn interner(&self) -> Interner {
                    Interner
                }

                fn try_fold_free_placeholder_const(
                    &mut self,
                    ty: chalk_ir::Ty<Interner>,
                    idx: chalk_ir::PlaceholderIndex,
                    outer_binder: DebruijnIndex,
                ) -> Result<chalk_ir::Const<Interner>, Self::Error> {
                    let x = from_placeholder_idx(self.db, idx);
                    let Some(idx) = self.generics.type_or_const_param_idx(x) else {
                        return Err(());
                    };
                    Ok(BoundVar::new(outer_binder, idx).to_const(Interner, ty))
                }

                fn try_fold_free_placeholder_ty(
                    &mut self,
                    idx: chalk_ir::PlaceholderIndex,
                    outer_binder: DebruijnIndex,
                ) -> std::result::Result<Ty, Self::Error> {
                    let x = from_placeholder_idx(self.db, idx);
                    let Some(idx) = self.generics.type_or_const_param_idx(x) else {
                        return Err(());
                    };
                    Ok(BoundVar::new(outer_binder, idx).to_ty(Interner))
                }
            }
            let filler = &mut Filler { db: ctx.db, generics: ctx.generics() };
            let result = ty.clone().try_fold_with(filler, DebruijnIndex::INNERMOST).unwrap_or(ty);
            make_binders(ctx.db, filler.generics, result)
        }
    }
}

impl InferenceContext<'_> {
    fn place_of_expr(&mut self, tgt_expr: ExprId) -> Option<HirPlace> {
        let r = self.place_of_expr_without_adjust(tgt_expr)?;
        let adjustments =
            self.result.expr_adjustments.get(&tgt_expr).map(|it| &**it).unwrap_or_default();
        apply_adjusts_to_place(&mut self.current_capture_span_stack, r, adjustments)
    }

    /// Pushes the span into `current_capture_span_stack`, *without clearing it first*.
    fn path_place(&mut self, path: &Path, id: ExprOrPatId) -> Option<HirPlace> {
        if path.type_anchor().is_some() {
            return None;
        }
        let hygiene = self.body.expr_or_pat_path_hygiene(id);
        self.resolver.resolve_path_in_value_ns_fully(self.db, path, hygiene).and_then(|result| {
            match result {
                ValueNs::LocalBinding(binding) => {
                    let mir_span = match id {
                        ExprOrPatId::ExprId(id) => MirSpan::ExprId(id),
                        ExprOrPatId::PatId(id) => MirSpan::PatId(id),
                    };
                    self.current_capture_span_stack.push(mir_span);
                    Some(HirPlace { local: binding, projections: Vec::new() })
                }
                _ => None,
            }
        })
    }

    /// Changes `current_capture_span_stack` to contain the stack of spans for this expr.
    fn place_of_expr_without_adjust(&mut self, tgt_expr: ExprId) -> Option<HirPlace> {
        self.current_capture_span_stack.clear();
        match &self.body[tgt_expr] {
            Expr::Path(p) => {
                let resolver_guard =
                    self.resolver.update_to_inner_scope(self.db, self.owner, tgt_expr);
                let result = self.path_place(p, tgt_expr.into());
                self.resolver.reset_to_guard(resolver_guard);
                return result;
            }
            Expr::Field { expr, name: _ } => {
                let mut place = self.place_of_expr(*expr)?;
                let field = self.result.field_resolution(tgt_expr)?;
                self.current_capture_span_stack.push(MirSpan::ExprId(tgt_expr));
                place.projections.push(ProjectionElem::Field(field));
                return Some(place);
            }
            Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
                if matches!(
                    self.expr_ty_after_adjustments(*expr).kind(Interner),
                    TyKind::Ref(..) | TyKind::Raw(..)
                ) {
                    let mut place = self.place_of_expr(*expr)?;
                    self.current_capture_span_stack.push(MirSpan::ExprId(tgt_expr));
                    place.projections.push(ProjectionElem::Deref);
                    return Some(place);
                }
            }
            _ => (),
        }
        None
    }

    fn push_capture(&mut self, place: HirPlace, kind: CaptureKind) {
        self.current_captures.push(CapturedItemWithoutTy {
            place,
            kind,
            span_stacks: smallvec![self.current_capture_span_stack.iter().copied().collect()],
        });
    }

    fn truncate_capture_spans(&self, capture: &mut CapturedItemWithoutTy, mut truncate_to: usize) {
        // The first span is the identifier, and it must always remain.
        truncate_to += 1;
        for span_stack in &mut capture.span_stacks {
            let mut remained = truncate_to;
            let mut actual_truncate_to = 0;
            for &span in &*span_stack {
                actual_truncate_to += 1;
                if !span.is_ref_span(self.body) {
                    remained -= 1;
                    if remained == 0 {
                        break;
                    }
                }
            }
            if actual_truncate_to < span_stack.len()
                && span_stack[actual_truncate_to].is_ref_span(self.body)
            {
                // Include the ref operator if there is one, we will fix it later (in `strip_captures_ref_span()`) if it's incorrect.
                actual_truncate_to += 1;
            }
            span_stack.truncate(actual_truncate_to);
        }
    }

    fn ref_expr(&mut self, expr: ExprId, place: Option<HirPlace>) {
        if let Some(place) = place {
            self.add_capture(place, CaptureKind::ByRef(BorrowKind::Shared));
        }
        self.walk_expr(expr);
    }

    fn add_capture(&mut self, place: HirPlace, kind: CaptureKind) {
        if self.is_upvar(&place) {
            self.push_capture(place, kind);
        }
    }

    fn mutate_path_pat(&mut self, path: &Path, id: PatId) {
        if let Some(place) = self.path_place(path, id.into()) {
            self.add_capture(
                place,
                CaptureKind::ByRef(BorrowKind::Mut { kind: MutBorrowKind::Default }),
            );
            self.current_capture_span_stack.pop(); // Remove the pattern span.
        }
    }

    fn mutate_expr(&mut self, expr: ExprId, place: Option<HirPlace>) {
        if let Some(place) = place {
            self.add_capture(
                place,
                CaptureKind::ByRef(BorrowKind::Mut { kind: MutBorrowKind::Default }),
            );
        }
        self.walk_expr(expr);
    }

    fn consume_expr(&mut self, expr: ExprId) {
        if let Some(place) = self.place_of_expr(expr) {
            self.consume_place(place);
        }
        self.walk_expr(expr);
    }

    fn consume_place(&mut self, place: HirPlace) {
        if self.is_upvar(&place) {
            let ty = place.ty(self);
            let kind = if self.is_ty_copy(ty) {
                CaptureKind::ByRef(BorrowKind::Shared)
            } else {
                CaptureKind::ByValue
            };
            self.push_capture(place, kind);
        }
    }

    fn walk_expr_with_adjust(&mut self, tgt_expr: ExprId, adjustment: &[Adjustment]) {
        if let Some((last, rest)) = adjustment.split_last() {
            match &last.kind {
                Adjust::NeverToAny | Adjust::Deref(None) | Adjust::Pointer(_) => {
                    self.walk_expr_with_adjust(tgt_expr, rest)
                }
                Adjust::Deref(Some(m)) => match m.0 {
                    Some(m) => {
                        self.ref_capture_with_adjusts(m, tgt_expr, rest);
                    }
                    None => unreachable!(),
                },
                Adjust::Borrow(b) => {
                    self.ref_capture_with_adjusts(b.mutability(), tgt_expr, rest);
                }
            }
        } else {
            self.walk_expr_without_adjust(tgt_expr);
        }
    }

    fn ref_capture_with_adjusts(&mut self, m: Mutability, tgt_expr: ExprId, rest: &[Adjustment]) {
        let capture_kind = match m {
            Mutability::Mut => CaptureKind::ByRef(BorrowKind::Mut { kind: MutBorrowKind::Default }),
            Mutability::Not => CaptureKind::ByRef(BorrowKind::Shared),
        };
        if let Some(place) = self.place_of_expr_without_adjust(tgt_expr) {
            if let Some(place) =
                apply_adjusts_to_place(&mut self.current_capture_span_stack, place, rest)
            {
                self.add_capture(place, capture_kind);
            }
        }
        self.walk_expr_with_adjust(tgt_expr, rest);
    }

    fn walk_expr(&mut self, tgt_expr: ExprId) {
        if let Some(it) = self.result.expr_adjustments.get_mut(&tgt_expr) {
            // FIXME: this take is completely unneeded, and just is here to make borrow checker
            // happy. Remove it if you can.
            let x_taken = mem::take(it);
            self.walk_expr_with_adjust(tgt_expr, &x_taken);
            *self.result.expr_adjustments.get_mut(&tgt_expr).unwrap() = x_taken;
        } else {
            self.walk_expr_without_adjust(tgt_expr);
        }
    }

    fn walk_expr_without_adjust(&mut self, tgt_expr: ExprId) {
        match &self.body[tgt_expr] {
            Expr::OffsetOf(_) => (),
            Expr::InlineAsm(e) => e.operands.iter().for_each(|(_, op)| match op {
                AsmOperand::In { expr, .. }
                | AsmOperand::Out { expr: Some(expr), .. }
                | AsmOperand::InOut { expr, .. } => self.walk_expr_without_adjust(*expr),
                AsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                    self.walk_expr_without_adjust(*in_expr);
                    if let Some(out_expr) = out_expr {
                        self.walk_expr_without_adjust(*out_expr);
                    }
                }
                AsmOperand::Out { expr: None, .. }
                | AsmOperand::Const(_)
                | AsmOperand::Label(_)
                | AsmOperand::Sym(_) => (),
            }),
            Expr::If { condition, then_branch, else_branch } => {
                self.consume_expr(*condition);
                self.consume_expr(*then_branch);
                if let &Some(expr) = else_branch {
                    self.consume_expr(expr);
                }
            }
            Expr::Async { statements, tail, .. }
            | Expr::Unsafe { statements, tail, .. }
            | Expr::Block { statements, tail, .. } => {
                for s in statements.iter() {
                    match s {
                        Statement::Let { pat, type_ref: _, initializer, else_branch } => {
                            if let Some(else_branch) = else_branch {
                                self.consume_expr(*else_branch);
                            }
                            if let Some(initializer) = initializer {
                                if else_branch.is_some() {
                                    self.consume_expr(*initializer);
                                } else {
                                    self.walk_expr(*initializer);
                                }
                                if let Some(place) = self.place_of_expr(*initializer) {
                                    self.consume_with_pat(place, *pat);
                                }
                            }
                        }
                        Statement::Expr { expr, has_semi: _ } => {
                            self.consume_expr(*expr);
                        }
                        Statement::Item(_) => (),
                    }
                }
                if let Some(tail) = tail {
                    self.consume_expr(*tail);
                }
            }
            Expr::Call { callee, args } => {
                self.consume_expr(*callee);
                self.consume_exprs(args.iter().copied());
            }
            Expr::MethodCall { receiver, args, .. } => {
                self.consume_expr(*receiver);
                self.consume_exprs(args.iter().copied());
            }
            Expr::Match { expr, arms } => {
                for arm in arms.iter() {
                    self.consume_expr(arm.expr);
                    if let Some(guard) = arm.guard {
                        self.consume_expr(guard);
                    }
                }
                self.walk_expr(*expr);
                if let Some(discr_place) = self.place_of_expr(*expr) {
                    if self.is_upvar(&discr_place) {
                        let mut capture_mode = None;
                        for arm in arms.iter() {
                            self.walk_pat(&mut capture_mode, arm.pat);
                        }
                        if let Some(c) = capture_mode {
                            self.push_capture(discr_place, c);
                        }
                    }
                }
            }
            Expr::Break { expr, label: _ }
            | Expr::Return { expr }
            | Expr::Yield { expr }
            | Expr::Yeet { expr } => {
                if let &Some(expr) = expr {
                    self.consume_expr(expr);
                }
            }
            &Expr::Become { expr } => {
                self.consume_expr(expr);
            }
            Expr::RecordLit { fields, spread, .. } => {
                if let &Some(expr) = spread {
                    self.consume_expr(expr);
                }
                self.consume_exprs(fields.iter().map(|it| it.expr));
            }
            Expr::Field { expr, name: _ } => self.select_from_expr(*expr),
            Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
                if matches!(
                    self.expr_ty_after_adjustments(*expr).kind(Interner),
                    TyKind::Ref(..) | TyKind::Raw(..)
                ) {
                    self.select_from_expr(*expr);
                } else if let Some((f, _)) = self.result.method_resolution(tgt_expr) {
                    let mutability = 'b: {
                        if let Some(deref_trait) =
                            self.resolve_lang_item(LangItem::DerefMut).and_then(|it| it.as_trait())
                        {
                            if let Some(deref_fn) = deref_trait
                                .trait_items(self.db)
                                .method_by_name(&Name::new_symbol_root(sym::deref_mut))
                            {
                                break 'b deref_fn == f;
                            }
                        }
                        false
                    };
                    let place = self.place_of_expr(*expr);
                    if mutability {
                        self.mutate_expr(*expr, place);
                    } else {
                        self.ref_expr(*expr, place);
                    }
                } else {
                    self.select_from_expr(*expr);
                }
            }
            Expr::Let { pat: _, expr } => {
                self.walk_expr(*expr);
                let place = self.place_of_expr(*expr);
                self.ref_expr(*expr, place);
            }
            Expr::UnaryOp { expr, op: _ }
            | Expr::Array(Array::Repeat { initializer: expr, repeat: _ })
            | Expr::Await { expr }
            | Expr::Loop { body: expr, label: _ }
            | Expr::Box { expr }
            | Expr::Cast { expr, type_ref: _ } => {
                self.consume_expr(*expr);
            }
            Expr::Ref { expr, rawness: _, mutability } => {
                // We need to do this before we push the span so the order will be correct.
                let place = self.place_of_expr(*expr);
                self.current_capture_span_stack.push(MirSpan::ExprId(tgt_expr));
                match mutability {
                    hir_def::type_ref::Mutability::Shared => self.ref_expr(*expr, place),
                    hir_def::type_ref::Mutability::Mut => self.mutate_expr(*expr, place),
                }
            }
            Expr::BinaryOp { lhs, rhs, op } => {
                let Some(op) = op else {
                    return;
                };
                if matches!(op, BinaryOp::Assignment { .. }) {
                    let place = self.place_of_expr(*lhs);
                    self.mutate_expr(*lhs, place);
                    self.consume_expr(*rhs);
                    return;
                }
                self.consume_expr(*lhs);
                self.consume_expr(*rhs);
            }
            Expr::Range { lhs, rhs, range_type: _ } => {
                if let &Some(expr) = lhs {
                    self.consume_expr(expr);
                }
                if let &Some(expr) = rhs {
                    self.consume_expr(expr);
                }
            }
            Expr::Index { base, index } => {
                self.select_from_expr(*base);
                self.consume_expr(*index);
            }
            Expr::Closure { .. } => {
                let ty = self.expr_ty(tgt_expr);
                let TyKind::Closure(id, _) = ty.kind(Interner) else {
                    never!("closure type is always closure");
                    return;
                };
                let (captures, _) =
                    self.result.closure_info.get(id).expect(
                        "We sort closures, so we should always have data for inner closures",
                    );
                let mut cc = mem::take(&mut self.current_captures);
                cc.extend(captures.iter().filter(|it| self.is_upvar(&it.place)).map(|it| {
                    CapturedItemWithoutTy {
                        place: it.place.clone(),
                        kind: it.kind,
                        span_stacks: it.span_stacks.clone(),
                    }
                }));
                self.current_captures = cc;
            }
            Expr::Array(Array::ElementList { elements: exprs }) | Expr::Tuple { exprs } => {
                self.consume_exprs(exprs.iter().copied())
            }
            &Expr::Assignment { target, value } => {
                self.walk_expr(value);
                let resolver_guard =
                    self.resolver.update_to_inner_scope(self.db, self.owner, tgt_expr);
                match self.place_of_expr(value) {
                    Some(rhs_place) => {
                        self.inside_assignment = true;
                        self.consume_with_pat(rhs_place, target);
                        self.inside_assignment = false;
                    }
                    None => self.body.walk_pats(target, &mut |pat| match &self.body[pat] {
                        Pat::Path(path) => self.mutate_path_pat(path, pat),
                        &Pat::Expr(expr) => {
                            let place = self.place_of_expr(expr);
                            self.mutate_expr(expr, place);
                        }
                        _ => {}
                    }),
                }
                self.resolver.reset_to_guard(resolver_guard);
            }

            Expr::Missing
            | Expr::Continue { .. }
            | Expr::Path(_)
            | Expr::Literal(_)
            | Expr::Const(_)
            | Expr::Underscore => (),
        }
    }

    fn walk_pat(&mut self, result: &mut Option<CaptureKind>, pat: PatId) {
        let mut update_result = |ck: CaptureKind| match result {
            Some(r) => {
                *r = cmp::max(*r, ck);
            }
            None => *result = Some(ck),
        };

        self.walk_pat_inner(
            pat,
            &mut update_result,
            BorrowKind::Mut { kind: MutBorrowKind::Default },
        );
    }

    fn walk_pat_inner(
        &mut self,
        p: PatId,
        update_result: &mut impl FnMut(CaptureKind),
        mut for_mut: BorrowKind,
    ) {
        match &self.body[p] {
            Pat::Ref { .. }
            | Pat::Box { .. }
            | Pat::Missing
            | Pat::Wild
            | Pat::Tuple { .. }
            | Pat::Expr(_)
            | Pat::Or(_) => (),
            Pat::TupleStruct { .. } | Pat::Record { .. } => {
                if let Some(variant) = self.result.variant_resolution_for_pat(p) {
                    let adt = variant.adt_id(self.db);
                    let is_multivariant = match adt {
                        hir_def::AdtId::EnumId(e) => e.enum_variants(self.db).variants.len() != 1,
                        _ => false,
                    };
                    if is_multivariant {
                        update_result(CaptureKind::ByRef(BorrowKind::Shared));
                    }
                }
            }
            Pat::Slice { .. }
            | Pat::ConstBlock(_)
            | Pat::Path(_)
            | Pat::Lit(_)
            | Pat::Range { .. } => {
                update_result(CaptureKind::ByRef(BorrowKind::Shared));
            }
            Pat::Bind { id, .. } => match self.result.binding_modes[p] {
                crate::BindingMode::Move => {
                    if self.is_ty_copy(self.result.type_of_binding[*id].clone()) {
                        update_result(CaptureKind::ByRef(BorrowKind::Shared));
                    } else {
                        update_result(CaptureKind::ByValue);
                    }
                }
                crate::BindingMode::Ref(r) => match r {
                    Mutability::Mut => update_result(CaptureKind::ByRef(for_mut)),
                    Mutability::Not => update_result(CaptureKind::ByRef(BorrowKind::Shared)),
                },
            },
        }
        if self.result.pat_adjustments.get(&p).is_some_and(|it| !it.is_empty()) {
            for_mut = BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture };
        }
        self.body.walk_pats_shallow(p, |p| self.walk_pat_inner(p, update_result, for_mut));
    }

    fn expr_ty(&self, expr: ExprId) -> Ty {
        self.result[expr].clone()
    }

    fn expr_ty_after_adjustments(&self, e: ExprId) -> Ty {
        let mut ty = None;
        if let Some(it) = self.result.expr_adjustments.get(&e) {
            if let Some(it) = it.last() {
                ty = Some(it.target.clone());
            }
        }
        ty.unwrap_or_else(|| self.expr_ty(e))
    }

    fn is_upvar(&self, place: &HirPlace) -> bool {
        if let Some(c) = self.current_closure {
            let InternedClosure(_, root) = self.db.lookup_intern_closure(c.into());
            return self.body.is_binding_upvar(place.local, root);
        }
        false
    }

    fn is_ty_copy(&mut self, ty: Ty) -> bool {
        if let TyKind::Closure(id, _) = ty.kind(Interner) {
            // FIXME: We handle closure as a special case, since chalk consider every closure as copy. We
            // should probably let chalk know which closures are copy, but I don't know how doing it
            // without creating query cycles.
            return self.result.closure_info.get(id).map(|it| it.1 == FnTrait::Fn).unwrap_or(true);
        }
        self.table.resolve_completely(ty).is_copy(self.db, self.owner)
    }

    fn select_from_expr(&mut self, expr: ExprId) {
        self.walk_expr(expr);
    }

    fn restrict_precision_for_unsafe(&mut self) {
        // FIXME: Borrow checker problems without this.
        let mut current_captures = std::mem::take(&mut self.current_captures);
        for capture in &mut current_captures {
            let mut ty = self.table.resolve_completely(self.result[capture.place.local].clone());
            if ty.as_raw_ptr().is_some() || ty.is_union() {
                capture.kind = CaptureKind::ByRef(BorrowKind::Shared);
                self.truncate_capture_spans(capture, 0);
                capture.place.projections.truncate(0);
                continue;
            }
            for (i, p) in capture.place.projections.iter().enumerate() {
                ty = p.projected_ty(
                    ty,
                    self.db,
                    |_, _, _| {
                        unreachable!("Closure field only happens in MIR");
                    },
                    self.owner.module(self.db).krate(),
                );
                if ty.as_raw_ptr().is_some() || ty.is_union() {
                    capture.kind = CaptureKind::ByRef(BorrowKind::Shared);
                    self.truncate_capture_spans(capture, i + 1);
                    capture.place.projections.truncate(i + 1);
                    break;
                }
            }
        }
        self.current_captures = current_captures;
    }

    fn adjust_for_move_closure(&mut self) {
        // FIXME: Borrow checker won't allow without this.
        let mut current_captures = std::mem::take(&mut self.current_captures);
        for capture in &mut current_captures {
            if let Some(first_deref) =
                capture.place.projections.iter().position(|proj| *proj == ProjectionElem::Deref)
            {
                self.truncate_capture_spans(capture, first_deref);
                capture.place.projections.truncate(first_deref);
            }
            capture.kind = CaptureKind::ByValue;
        }
        self.current_captures = current_captures;
    }

    fn minimize_captures(&mut self) {
        self.current_captures.sort_unstable_by_key(|it| it.place.projections.len());
        let mut hash_map = FxHashMap::<HirPlace, usize>::default();
        let result = mem::take(&mut self.current_captures);
        for mut item in result {
            let mut lookup_place = HirPlace { local: item.place.local, projections: vec![] };
            let mut it = item.place.projections.iter();
            let prev_index = loop {
                if let Some(k) = hash_map.get(&lookup_place) {
                    break Some(*k);
                }
                match it.next() {
                    Some(it) => {
                        lookup_place.projections.push(it.clone());
                    }
                    None => break None,
                }
            };
            match prev_index {
                Some(p) => {
                    let prev_projections_len = self.current_captures[p].place.projections.len();
                    self.truncate_capture_spans(&mut item, prev_projections_len);
                    self.current_captures[p].span_stacks.extend(item.span_stacks);
                    let len = self.current_captures[p].place.projections.len();
                    let kind_after_truncate =
                        item.place.capture_kind_of_truncated_place(item.kind, len);
                    self.current_captures[p].kind =
                        cmp::max(kind_after_truncate, self.current_captures[p].kind);
                }
                None => {
                    hash_map.insert(item.place.clone(), self.current_captures.len());
                    self.current_captures.push(item);
                }
            }
        }
    }

    fn consume_with_pat(&mut self, mut place: HirPlace, tgt_pat: PatId) {
        let adjustments_count =
            self.result.pat_adjustments.get(&tgt_pat).map(|it| it.len()).unwrap_or_default();
        place.projections.extend((0..adjustments_count).map(|_| ProjectionElem::Deref));
        self.current_capture_span_stack
            .extend((0..adjustments_count).map(|_| MirSpan::PatId(tgt_pat)));
        'reset_span_stack: {
            match &self.body[tgt_pat] {
                Pat::Missing | Pat::Wild => (),
                Pat::Tuple { args, ellipsis } => {
                    let (al, ar) = args.split_at(ellipsis.map_or(args.len(), |it| it as usize));
                    let field_count = match self.result[tgt_pat].kind(Interner) {
                        TyKind::Tuple(_, s) => s.len(Interner),
                        _ => break 'reset_span_stack,
                    };
                    let fields = 0..field_count;
                    let it = al.iter().zip(fields.clone()).chain(ar.iter().rev().zip(fields.rev()));
                    for (&arg, i) in it {
                        let mut p = place.clone();
                        self.current_capture_span_stack.push(MirSpan::PatId(arg));
                        p.projections.push(ProjectionElem::Field(Either::Right(TupleFieldId {
                            tuple: TupleId(!0), // dummy this, as its unused anyways
                            index: i as u32,
                        })));
                        self.consume_with_pat(p, arg);
                        self.current_capture_span_stack.pop();
                    }
                }
                Pat::Or(pats) => {
                    for pat in pats.iter() {
                        self.consume_with_pat(place.clone(), *pat);
                    }
                }
                Pat::Record { args, .. } => {
                    let Some(variant) = self.result.variant_resolution_for_pat(tgt_pat) else {
                        break 'reset_span_stack;
                    };
                    match variant {
                        VariantId::EnumVariantId(_) | VariantId::UnionId(_) => {
                            self.consume_place(place)
                        }
                        VariantId::StructId(s) => {
                            let vd = s.fields(self.db);
                            for field_pat in args.iter() {
                                let arg = field_pat.pat;
                                let Some(local_id) = vd.field(&field_pat.name) else {
                                    continue;
                                };
                                let mut p = place.clone();
                                self.current_capture_span_stack.push(MirSpan::PatId(arg));
                                p.projections.push(ProjectionElem::Field(Either::Left(FieldId {
                                    parent: variant,
                                    local_id,
                                })));
                                self.consume_with_pat(p, arg);
                                self.current_capture_span_stack.pop();
                            }
                        }
                    }
                }
                Pat::Range { .. } | Pat::Slice { .. } | Pat::ConstBlock(_) | Pat::Lit(_) => {
                    self.consume_place(place)
                }
                Pat::Path(path) => {
                    if self.inside_assignment {
                        self.mutate_path_pat(path, tgt_pat);
                    }
                    self.consume_place(place);
                }
                &Pat::Bind { id, subpat: _ } => {
                    let mode = self.result.binding_modes[tgt_pat];
                    let capture_kind = match mode {
                        BindingMode::Move => {
                            self.consume_place(place);
                            break 'reset_span_stack;
                        }
                        BindingMode::Ref(Mutability::Not) => BorrowKind::Shared,
                        BindingMode::Ref(Mutability::Mut) => {
                            BorrowKind::Mut { kind: MutBorrowKind::Default }
                        }
                    };
                    self.current_capture_span_stack.push(MirSpan::BindingId(id));
                    self.add_capture(place, CaptureKind::ByRef(capture_kind));
                    self.current_capture_span_stack.pop();
                }
                Pat::TupleStruct { path: _, args, ellipsis } => {
                    let Some(variant) = self.result.variant_resolution_for_pat(tgt_pat) else {
                        break 'reset_span_stack;
                    };
                    match variant {
                        VariantId::EnumVariantId(_) | VariantId::UnionId(_) => {
                            self.consume_place(place)
                        }
                        VariantId::StructId(s) => {
                            let vd = s.fields(self.db);
                            let (al, ar) =
                                args.split_at(ellipsis.map_or(args.len(), |it| it as usize));
                            let fields = vd.fields().iter();
                            let it = al
                                .iter()
                                .zip(fields.clone())
                                .chain(ar.iter().rev().zip(fields.rev()));
                            for (&arg, (i, _)) in it {
                                let mut p = place.clone();
                                self.current_capture_span_stack.push(MirSpan::PatId(arg));
                                p.projections.push(ProjectionElem::Field(Either::Left(FieldId {
                                    parent: variant,
                                    local_id: i,
                                })));
                                self.consume_with_pat(p, arg);
                                self.current_capture_span_stack.pop();
                            }
                        }
                    }
                }
                Pat::Ref { pat, mutability: _ } => {
                    self.current_capture_span_stack.push(MirSpan::PatId(tgt_pat));
                    place.projections.push(ProjectionElem::Deref);
                    self.consume_with_pat(place, *pat);
                    self.current_capture_span_stack.pop();
                }
                Pat::Box { .. } => (), // not supported
                &Pat::Expr(expr) => {
                    self.consume_place(place);
                    let pat_capture_span_stack = mem::take(&mut self.current_capture_span_stack);
                    let old_inside_assignment = mem::replace(&mut self.inside_assignment, false);
                    let lhs_place = self.place_of_expr(expr);
                    self.mutate_expr(expr, lhs_place);
                    self.inside_assignment = old_inside_assignment;
                    self.current_capture_span_stack = pat_capture_span_stack;
                }
            }
        }
        self.current_capture_span_stack
            .truncate(self.current_capture_span_stack.len() - adjustments_count);
    }

    fn consume_exprs(&mut self, exprs: impl Iterator<Item = ExprId>) {
        for expr in exprs {
            self.consume_expr(expr);
        }
    }

    fn closure_kind(&self) -> FnTrait {
        let mut r = FnTrait::Fn;
        for it in &self.current_captures {
            r = cmp::min(
                r,
                match &it.kind {
                    CaptureKind::ByRef(BorrowKind::Mut { .. }) => FnTrait::FnMut,
                    CaptureKind::ByRef(BorrowKind::Shallow | BorrowKind::Shared) => FnTrait::Fn,
                    CaptureKind::ByValue => FnTrait::FnOnce,
                },
            )
        }
        r
    }

    fn analyze_closure(&mut self, closure: ClosureId) -> FnTrait {
        let InternedClosure(_, root) = self.db.lookup_intern_closure(closure.into());
        self.current_closure = Some(closure);
        let Expr::Closure { body, capture_by, .. } = &self.body[root] else {
            unreachable!("Closure expression id is always closure");
        };
        self.consume_expr(*body);
        for item in &self.current_captures {
            if matches!(
                item.kind,
                CaptureKind::ByRef(BorrowKind::Mut {
                    kind: MutBorrowKind::Default | MutBorrowKind::TwoPhasedBorrow
                })
            ) && !item.place.projections.contains(&ProjectionElem::Deref)
            {
                // FIXME: remove the `mutated_bindings_in_closure` completely and add proper fake reads in
                // MIR. I didn't do that due duplicate diagnostics.
                self.result.mutated_bindings_in_closure.insert(item.place.local);
            }
        }
        self.restrict_precision_for_unsafe();
        // `closure_kind` should be done before adjust_for_move_closure
        // If there exists pre-deduced kind of a closure, use it instead of one determined by capture, as rustc does.
        // rustc also does diagnostics here if the latter is not a subtype of the former.
        let closure_kind = self
            .result
            .closure_info
            .get(&closure)
            .map_or_else(|| self.closure_kind(), |info| info.1);
        match capture_by {
            CaptureBy::Value => self.adjust_for_move_closure(),
            CaptureBy::Ref => (),
        }
        self.minimize_captures();
        self.strip_captures_ref_span();
        let result = mem::take(&mut self.current_captures);
        let captures = result.into_iter().map(|it| it.with_ty(self)).collect::<Vec<_>>();
        self.result.closure_info.insert(closure, (captures, closure_kind));
        closure_kind
    }

    fn strip_captures_ref_span(&mut self) {
        // FIXME: Borrow checker won't allow without this.
        let mut captures = std::mem::take(&mut self.current_captures);
        for capture in &mut captures {
            if matches!(capture.kind, CaptureKind::ByValue) {
                for span_stack in &mut capture.span_stacks {
                    if span_stack[span_stack.len() - 1].is_ref_span(self.body) {
                        span_stack.truncate(span_stack.len() - 1);
                    }
                }
            }
        }
        self.current_captures = captures;
    }

    pub(crate) fn infer_closures(&mut self) {
        let deferred_closures = self.sort_closures();
        for (closure, exprs) in deferred_closures.into_iter().rev() {
            self.current_captures = vec![];
            let kind = self.analyze_closure(closure);

            for (derefed_callee, callee_ty, params, expr) in exprs {
                if let &Expr::Call { callee, .. } = &self.body[expr] {
                    let mut adjustments =
                        self.result.expr_adjustments.remove(&callee).unwrap_or_default().into_vec();
                    self.write_fn_trait_method_resolution(
                        kind,
                        &derefed_callee,
                        &mut adjustments,
                        &callee_ty,
                        &params,
                        expr,
                    );
                    self.result.expr_adjustments.insert(callee, adjustments.into_boxed_slice());
                }
            }
        }
    }

    /// We want to analyze some closures before others, to have a correct analysis:
    /// * We should analyze nested closures before the parent, since the parent should capture some of
    ///   the things that its children captures.
    /// * If a closure calls another closure, we need to analyze the callee, to find out how we should
    ///   capture it (e.g. by move for FnOnce)
    ///
    /// These dependencies are collected in the main inference. We do a topological sort in this function. It
    /// will consume the `deferred_closures` field and return its content in a sorted vector.
    fn sort_closures(&mut self) -> Vec<(ClosureId, Vec<(Ty, Ty, Vec<Ty>, ExprId)>)> {
        let mut deferred_closures = mem::take(&mut self.deferred_closures);
        let mut dependents_count: FxHashMap<ClosureId, usize> =
            deferred_closures.keys().map(|it| (*it, 0)).collect();
        for deps in self.closure_dependencies.values() {
            for dep in deps {
                *dependents_count.entry(*dep).or_default() += 1;
            }
        }
        let mut queue: Vec<_> =
            deferred_closures.keys().copied().filter(|it| dependents_count[it] == 0).collect();
        let mut result = vec![];
        while let Some(it) = queue.pop() {
            if let Some(d) = deferred_closures.remove(&it) {
                result.push((it, d));
            }
            for dep in self.closure_dependencies.get(&it).into_iter().flat_map(|it| it.iter()) {
                let cnt = dependents_count.get_mut(dep).unwrap();
                *cnt -= 1;
                if *cnt == 0 {
                    queue.push(*dep);
                }
            }
        }
        assert!(deferred_closures.is_empty(), "we should have analyzed all closures");
        result
    }

    pub(super) fn add_current_closure_dependency(&mut self, dep: ClosureId) {
        if let Some(c) = self.current_closure {
            if !dep_creates_cycle(&self.closure_dependencies, &mut FxHashSet::default(), c, dep) {
                self.closure_dependencies.entry(c).or_default().push(dep);
            }
        }

        fn dep_creates_cycle(
            closure_dependencies: &FxHashMap<ClosureId, Vec<ClosureId>>,
            visited: &mut FxHashSet<ClosureId>,
            from: ClosureId,
            to: ClosureId,
        ) -> bool {
            if !visited.insert(from) {
                return false;
            }

            if from == to {
                return true;
            }

            if let Some(deps) = closure_dependencies.get(&to) {
                for dep in deps {
                    if dep_creates_cycle(closure_dependencies, visited, from, *dep) {
                        return true;
                    }
                }
            }

            false
        }
    }
}

/// Call this only when the last span in the stack isn't a split.
fn apply_adjusts_to_place(
    current_capture_span_stack: &mut Vec<MirSpan>,
    mut r: HirPlace,
    adjustments: &[Adjustment],
) -> Option<HirPlace> {
    let span = *current_capture_span_stack.last().expect("empty capture span stack");
    for adj in adjustments {
        match &adj.kind {
            Adjust::Deref(None) => {
                current_capture_span_stack.push(span);
                r.projections.push(ProjectionElem::Deref);
            }
            _ => return None,
        }
    }
    Some(r)
}
