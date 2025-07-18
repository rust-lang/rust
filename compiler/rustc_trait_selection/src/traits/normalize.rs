//! Deeply normalize types using the old trait solver.

use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_hir::def::DefKind;
use rustc_infer::infer::at::At;
use rustc_infer::infer::{InferCtxt, InferOk};
use rustc_infer::traits::{
    FromSolverError, Normalized, Obligation, PredicateObligations, TraitEngine,
};
use rustc_macros::extension;
use rustc_middle::span_bug;
use rustc_middle::traits::{ObligationCause, ObligationCauseCode};
use rustc_middle::ty::{
    self, AliasTerm, Term, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitable,
    TypeVisitableExt, TypingMode,
};
use tracing::{debug, instrument};

use super::{BoundVarReplacer, PlaceholderReplacer, SelectionContext, project};
use crate::error_reporting::InferCtxtErrorExt;
use crate::error_reporting::traits::OverflowCause;
use crate::solve::NextSolverError;

#[extension(pub trait NormalizeExt<'tcx>)]
impl<'tcx> At<'_, 'tcx> {
    /// Normalize a value using the `AssocTypeNormalizer`.
    ///
    /// This normalization should be used when the type contains inference variables or the
    /// projection may be fallible.
    fn normalize<T: TypeFoldable<TyCtxt<'tcx>>>(&self, value: T) -> InferOk<'tcx, T> {
        if self.infcx.next_trait_solver() {
            InferOk { value, obligations: PredicateObligations::new() }
        } else {
            let mut selcx = SelectionContext::new(self.infcx);
            let Normalized { value, obligations } =
                normalize_with_depth(&mut selcx, self.param_env, self.cause.clone(), 0, value);
            InferOk { value, obligations }
        }
    }

    /// Deeply normalizes `value`, replacing all aliases which can by normalized in
    /// the current environment. In the new solver this errors in case normalization
    /// fails or is ambiguous.
    ///
    /// In the old solver this simply uses `normalizes` and adds the nested obligations
    /// to the `fulfill_cx`. This is necessary as we otherwise end up recomputing the
    /// same goals in both a temporary and the shared context which negatively impacts
    /// performance as these don't share caching.
    ///
    /// FIXME(-Znext-solver=no): For performance reasons, we currently reuse an existing
    /// fulfillment context in the old solver. Once we have removed the old solver, we
    /// can remove the `fulfill_cx` parameter on this function.
    fn deeply_normalize<T, E>(
        self,
        value: T,
        fulfill_cx: &mut dyn TraitEngine<'tcx, E>,
    ) -> Result<T, Vec<E>>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
        E: FromSolverError<'tcx, NextSolverError<'tcx>>,
    {
        if self.infcx.next_trait_solver() {
            crate::solve::deeply_normalize(self, value)
        } else {
            if fulfill_cx.has_pending_obligations() {
                let pending_obligations = fulfill_cx.pending_obligations();
                span_bug!(
                    pending_obligations[0].cause.span,
                    "deeply_normalize should not be called with pending obligations: \
                    {pending_obligations:#?}"
                );
            }
            let value = self
                .normalize(value)
                .into_value_registering_obligations(self.infcx, &mut *fulfill_cx);
            let errors = fulfill_cx.select_all_or_error(self.infcx);
            let value = self.infcx.resolve_vars_if_possible(value);
            if errors.is_empty() {
                Ok(value)
            } else {
                // Drop pending obligations, since deep normalization may happen
                // in a loop and we don't want to trigger the assertion on the next
                // iteration due to pending ambiguous obligations we've left over.
                let _ = fulfill_cx.collect_remaining_errors(self.infcx);
                Err(errors)
            }
        }
    }
}

/// As `normalize`, but with a custom depth.
pub(crate) fn normalize_with_depth<'a, 'b, 'tcx, T>(
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    value: T,
) -> Normalized<'tcx, T>
where
    T: TypeFoldable<TyCtxt<'tcx>>,
{
    let mut obligations = PredicateObligations::new();
    let value = normalize_with_depth_to(selcx, param_env, cause, depth, value, &mut obligations);
    Normalized { value, obligations }
}

#[instrument(level = "info", skip(selcx, param_env, cause, obligations))]
pub(crate) fn normalize_with_depth_to<'a, 'b, 'tcx, T>(
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    cause: ObligationCause<'tcx>,
    depth: usize,
    value: T,
    obligations: &mut PredicateObligations<'tcx>,
) -> T
where
    T: TypeFoldable<TyCtxt<'tcx>>,
{
    debug!(obligations.len = obligations.len());
    let mut normalizer = AssocTypeNormalizer::new(selcx, param_env, cause, depth, obligations);
    let result = ensure_sufficient_stack(|| AssocTypeNormalizer::fold(&mut normalizer, value));
    debug!(?result, obligations.len = normalizer.obligations.len());
    debug!(?normalizer.obligations,);
    result
}

pub(super) fn needs_normalization<'tcx, T: TypeVisitable<TyCtxt<'tcx>>>(
    infcx: &InferCtxt<'tcx>,
    value: &T,
) -> bool {
    let mut flags = ty::TypeFlags::HAS_ALIAS;

    // Opaques are treated as rigid outside of `TypingMode::PostAnalysis`,
    // so we can ignore those.
    match infcx.typing_mode() {
        // FIXME(#132279): We likely want to reveal opaques during post borrowck analysis
        TypingMode::Coherence
        | TypingMode::Analysis { .. }
        | TypingMode::Borrowck { .. }
        | TypingMode::PostBorrowckAnalysis { .. } => flags.remove(ty::TypeFlags::HAS_TY_OPAQUE),
        TypingMode::PostAnalysis => {}
    }

    value.has_type_flags(flags)
}

struct AssocTypeNormalizer<'a, 'b, 'tcx> {
    selcx: &'a mut SelectionContext<'b, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    cause: ObligationCause<'tcx>,
    obligations: &'a mut PredicateObligations<'tcx>,
    depth: usize,
    universes: Vec<Option<ty::UniverseIndex>>,
}

impl<'a, 'b, 'tcx> AssocTypeNormalizer<'a, 'b, 'tcx> {
    fn new(
        selcx: &'a mut SelectionContext<'b, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        cause: ObligationCause<'tcx>,
        depth: usize,
        obligations: &'a mut PredicateObligations<'tcx>,
    ) -> AssocTypeNormalizer<'a, 'b, 'tcx> {
        debug_assert!(!selcx.infcx.next_trait_solver());
        AssocTypeNormalizer { selcx, param_env, cause, obligations, depth, universes: vec![] }
    }

    fn fold<T: TypeFoldable<TyCtxt<'tcx>>>(&mut self, value: T) -> T {
        let value = self.selcx.infcx.resolve_vars_if_possible(value);
        debug!(?value);

        assert!(
            !value.has_escaping_bound_vars(),
            "Normalizing {value:?} without wrapping in a `Binder`"
        );

        if !needs_normalization(self.selcx.infcx, &value) { value } else { value.fold_with(self) }
    }

    // FIXME(mgca): While this supports constants, it is only used for types by default right now
    #[instrument(level = "debug", skip(self), ret)]
    fn normalize_trait_projection(&mut self, proj: AliasTerm<'tcx>) -> Term<'tcx> {
        if !proj.has_escaping_bound_vars() {
            // When we don't have escaping bound vars we can normalize ambig aliases
            // to inference variables (done in `normalize_projection_ty`). This would
            // be wrong if there were escaping bound vars as even if we instantiated
            // the bound vars with placeholders, we wouldn't be able to map them back
            // after normalization succeeded.
            //
            // Also, as an optimization: when we don't have escaping bound vars, we don't
            // need to replace them with placeholders (see branch below).
            let proj = proj.fold_with(self);
            project::normalize_projection_term(
                self.selcx,
                self.param_env,
                proj,
                self.cause.clone(),
                self.depth,
                self.obligations,
            )
        } else {
            // If there are escaping bound vars, we temporarily replace the
            // bound vars with placeholders. Note though, that in the case
            // that we still can't project for whatever reason (e.g. self
            // type isn't known enough), we *can't* register an obligation
            // and return an inference variable (since then that obligation
            // would have bound vars and that's a can of worms). Instead,
            // we just give up and fall back to pretending like we never tried!
            //
            // Note: this isn't necessarily the final approach here; we may
            // want to figure out how to register obligations with escaping vars
            // or handle this some other way.
            let infcx = self.selcx.infcx;
            let (proj, mapped_regions, mapped_types, mapped_consts) =
                BoundVarReplacer::replace_bound_vars(infcx, &mut self.universes, proj);
            let proj = proj.fold_with(self);
            let normalized_term = project::opt_normalize_projection_term(
                self.selcx,
                self.param_env,
                proj,
                self.cause.clone(),
                self.depth,
                self.obligations,
            )
            .ok()
            .flatten()
            .unwrap_or_else(|| proj.to_term(infcx.tcx));

            PlaceholderReplacer::replace_placeholders(
                infcx,
                mapped_regions,
                mapped_types,
                mapped_consts,
                &self.universes,
                normalized_term,
            )
        }
    }

    // FIXME(mgca): While this supports constants, it is only used for types by default right now
    #[instrument(level = "debug", skip(self), ret)]
    fn normalize_inherent_projection(&mut self, inherent: AliasTerm<'tcx>) -> Term<'tcx> {
        if !inherent.has_escaping_bound_vars() {
            // When we don't have escaping bound vars we can normalize ambig aliases
            // to inference variables (done in `normalize_projection_ty`). This would
            // be wrong if there were escaping bound vars as even if we instantiated
            // the bound vars with placeholders, we wouldn't be able to map them back
            // after normalization succeeded.
            //
            // Also, as an optimization: when we don't have escaping bound vars, we don't
            // need to replace them with placeholders (see branch below).

            let inherent = inherent.fold_with(self);
            project::normalize_inherent_projection(
                self.selcx,
                self.param_env,
                inherent,
                self.cause.clone(),
                self.depth,
                self.obligations,
            )
        } else {
            let infcx = self.selcx.infcx;
            let (inherent, mapped_regions, mapped_types, mapped_consts) =
                BoundVarReplacer::replace_bound_vars(infcx, &mut self.universes, inherent);
            let inherent = inherent.fold_with(self);
            let inherent = project::normalize_inherent_projection(
                self.selcx,
                self.param_env,
                inherent,
                self.cause.clone(),
                self.depth,
                self.obligations,
            );

            PlaceholderReplacer::replace_placeholders(
                infcx,
                mapped_regions,
                mapped_types,
                mapped_consts,
                &self.universes,
                inherent,
            )
        }
    }

    // FIXME(mgca): While this supports constants, it is only used for types by default right now
    #[instrument(level = "debug", skip(self), ret)]
    fn normalize_free_alias(&mut self, free: AliasTerm<'tcx>) -> Term<'tcx> {
        let recursion_limit = self.cx().recursion_limit();
        if !recursion_limit.value_within_limit(self.depth) {
            self.selcx.infcx.err_ctxt().report_overflow_error(
                OverflowCause::DeeplyNormalize(free.into()),
                self.cause.span,
                false,
                |diag| {
                    diag.note(crate::fluent_generated::trait_selection_ty_alias_overflow);
                },
            );
        }

        // We don't replace bound vars in the generic arguments of the free alias with
        // placeholders. This doesn't cause any issues as instantiating parameters with
        // bound variables is special-cased to rewrite the debruijn index to be higher
        // whenever we fold through a binder.
        //
        // However, we do replace any escaping bound vars in the resulting goals with
        // placeholders as the trait solver does not expect to encounter escaping bound
        // vars in obligations.
        //
        // FIXME(lazy_type_alias): Check how much this actually matters for perf before
        // stabilization. This is a bit weird and generally not how we handle binders in
        // the compiler so ideally we'd do the same boundvar->placeholder->boundvar dance
        // that other kinds of normalization do.
        let infcx = self.selcx.infcx;
        self.obligations.extend(
            infcx.tcx.predicates_of(free.def_id).instantiate_own(infcx.tcx, free.args).map(
                |(mut predicate, span)| {
                    if free.has_escaping_bound_vars() {
                        (predicate, ..) = BoundVarReplacer::replace_bound_vars(
                            infcx,
                            &mut self.universes,
                            predicate,
                        );
                    }
                    let mut cause = self.cause.clone();
                    cause.map_code(|code| ObligationCauseCode::TypeAlias(code, span, free.def_id));
                    Obligation::new(infcx.tcx, cause, self.param_env, predicate)
                },
            ),
        );
        self.depth += 1;
        let res = if free.kind(infcx.tcx).is_type() {
            infcx.tcx.type_of(free.def_id).instantiate(infcx.tcx, free.args).fold_with(self).into()
        } else {
            // FIXME(mgca): once const items are actual aliases defined as equal to type system consts
            // this should instead use that rather than evaluating.
            super::evaluate_const(infcx, free.to_term(infcx.tcx).expect_const(), self.param_env)
                .super_fold_with(self)
                .into()
        };
        self.depth -= 1;
        res
    }
}

impl<'a, 'b, 'tcx> TypeFolder<TyCtxt<'tcx>> for AssocTypeNormalizer<'a, 'b, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.selcx.tcx()
    }

    fn fold_binder<T: TypeFoldable<TyCtxt<'tcx>>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        self.universes.push(None);
        let t = t.super_fold_with(self);
        self.universes.pop();
        t
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !needs_normalization(self.selcx.infcx, &ty) {
            return ty;
        }

        let (kind, data) = match *ty.kind() {
            ty::Alias(kind, data) => (kind, data),
            _ => return ty.super_fold_with(self),
        };

        // We try to be a little clever here as a performance optimization in
        // cases where there are nested projections under binders.
        // For example:
        // ```
        // for<'a> fn(<T as Foo>::One<'a, Box<dyn Bar<'a, Item=<T as Foo>::Two<'a>>>>)
        // ```
        // We normalize the args on the projection before the projecting, but
        // if we're naive, we'll
        //   replace bound vars on inner, project inner, replace placeholders on inner,
        //   replace bound vars on outer, project outer, replace placeholders on outer
        //
        // However, if we're a bit more clever, we can replace the bound vars
        // on the entire type before normalizing nested projections, meaning we
        //   replace bound vars on outer, project inner,
        //   project outer, replace placeholders on outer
        //
        // This is possible because the inner `'a` will already be a placeholder
        // when we need to normalize the inner projection
        //
        // On the other hand, this does add a bit of complexity, since we only
        // replace bound vars if the current type is a `Projection` and we need
        // to make sure we don't forget to fold the args regardless.

        match kind {
            ty::Opaque => {
                // Only normalize `impl Trait` outside of type inference, usually in codegen.
                match self.selcx.infcx.typing_mode() {
                    // FIXME(#132279): We likely want to reveal opaques during post borrowck analysis
                    TypingMode::Coherence
                    | TypingMode::Analysis { .. }
                    | TypingMode::Borrowck { .. }
                    | TypingMode::PostBorrowckAnalysis { .. } => ty.super_fold_with(self),
                    TypingMode::PostAnalysis => {
                        let recursion_limit = self.cx().recursion_limit();
                        if !recursion_limit.value_within_limit(self.depth) {
                            self.selcx.infcx.err_ctxt().report_overflow_error(
                                OverflowCause::DeeplyNormalize(data.into()),
                                self.cause.span,
                                true,
                                |_| {},
                            );
                        }

                        let args = data.args.fold_with(self);
                        let generic_ty = self.cx().type_of(data.def_id);
                        let concrete_ty = generic_ty.instantiate(self.cx(), args);
                        self.depth += 1;
                        let folded_ty = self.fold_ty(concrete_ty);
                        self.depth -= 1;
                        folded_ty
                    }
                }
            }

            ty::Projection => self.normalize_trait_projection(data.into()).expect_type(),
            ty::Inherent => self.normalize_inherent_projection(data.into()).expect_type(),
            ty::Free => self.normalize_free_alias(data.into()).expect_type(),
        }
    }

    #[instrument(skip(self), level = "debug")]
    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        let tcx = self.selcx.tcx();
        if tcx.features().generic_const_exprs() || !needs_normalization(self.selcx.infcx, &ct) {
            return ct;
        }

        // Doing "proper" normalization of const aliases is inherently cyclic until const items
        // are real aliases instead of having bodies. We gate proper const alias handling behind
        // mgca to avoid breaking stable code, though this should become the "main" codepath long
        // before mgca is stabilized.
        //
        // FIXME(BoxyUwU): Enabling this by default is blocked on a refactoring to how const items
        // are represented.
        if tcx.features().min_generic_const_args() {
            let uv = match ct.kind() {
                ty::ConstKind::Unevaluated(uv) => uv,
                _ => return ct.super_fold_with(self),
            };

            let ct = match tcx.def_kind(uv.def) {
                DefKind::AssocConst => match tcx.def_kind(tcx.parent(uv.def)) {
                    DefKind::Trait => self.normalize_trait_projection(uv.into()),
                    DefKind::Impl { of_trait: false } => {
                        self.normalize_inherent_projection(uv.into())
                    }
                    kind => unreachable!(
                        "unexpected `DefKind` for const alias' resolution's parent def: {:?}",
                        kind
                    ),
                },
                DefKind::Const | DefKind::AnonConst => self.normalize_free_alias(uv.into()),
                kind => {
                    unreachable!("unexpected `DefKind` for const alias to resolve to: {:?}", kind)
                }
            };

            // We re-fold the normalized const as the `ty` field on `ConstKind::Value` may be
            // unnormalized after const evaluation returns.
            ct.expect_const().super_fold_with(self)
        } else {
            let ct = ct.super_fold_with(self);
            return super::with_replaced_escaping_bound_vars(
                self.selcx.infcx,
                &mut self.universes,
                ct,
                |ct| super::evaluate_const(self.selcx.infcx, ct, self.param_env),
            )
            .super_fold_with(self);
            // We re-fold the normalized const as the `ty` field on `ConstKind::Value` may be
            // unnormalized after const evaluation returns.
        }
    }

    #[inline]
    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        if p.allow_normalization() && needs_normalization(self.selcx.infcx, &p) {
            p.super_fold_with(self)
        } else {
            p
        }
    }
}
