//! Code for the 'normalization' query. This consists of a wrapper
//! which folds deeply, invoking the underlying
//! `normalize_canonicalized_projection_ty` query when it encounters projections.

use rustc_data_structures::sso::SsoHashMap;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_infer::traits::PredicateObligations;
use rustc_macros::extension;
pub use rustc_middle::traits::query::NormalizationResult;
use rustc_middle::ty::{
    self, FallibleTypeFolder, Ty, TyCtxt, TypeFoldable, TypeSuperFoldable, TypeSuperVisitable,
    TypeVisitable, TypeVisitableExt, TypeVisitor, TypingMode,
};
use rustc_span::DUMMY_SP;
use tracing::{debug, info, instrument};

use super::NoSolution;
use crate::error_reporting::InferCtxtErrorExt;
use crate::error_reporting::traits::OverflowCause;
use crate::infer::at::At;
use crate::infer::canonical::OriginalQueryValues;
use crate::infer::{InferCtxt, InferOk};
use crate::traits::normalize::needs_normalization;
use crate::traits::{
    BoundVarReplacer, Normalized, ObligationCause, PlaceholderReplacer, ScrubbedTraitError,
};

#[extension(pub trait QueryNormalizeExt<'tcx>)]
impl<'a, 'tcx> At<'a, 'tcx> {
    /// Normalize `value` in the context of the inference context,
    /// yielding a resulting type, or an error if `value` cannot be
    /// normalized. If you don't care about regions, you should prefer
    /// `normalize_erasing_regions`, which is more efficient.
    ///
    /// If the normalization succeeds, returns back the normalized
    /// value along with various outlives relations (in the form of
    /// obligations that must be discharged).
    ///
    /// This normalization should *only* be used when the projection is well-formed and
    /// does not have possible ambiguity (contains inference variables).
    ///
    /// After codegen, when lifetimes do not matter, it is preferable to instead
    /// use [`TyCtxt::normalize_erasing_regions`], which wraps this procedure.
    ///
    /// N.B. Once the new solver is stabilized this method of normalization will
    /// likely be removed as trait solver operations are already cached by the query
    /// system making this redundant.
    fn query_normalize<T>(self, value: T) -> Result<Normalized<'tcx, T>, NoSolution>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        debug!(
            "normalize::<{}>(value={:?}, param_env={:?}, cause={:?})",
            std::any::type_name::<T>(),
            value,
            self.param_env,
            self.cause,
        );

        // This is actually a consequence by the way `normalize_erasing_regions` works currently.
        // Because it needs to call the `normalize_generic_arg_after_erasing_regions`, it folds
        // through tys and consts in a `TypeFoldable`. Importantly, it skips binders, leaving us
        // with trying to normalize with escaping bound vars.
        //
        // Here, we just add the universes that we *would* have created had we passed through the binders.
        //
        // We *could* replace escaping bound vars eagerly here, but it doesn't seem really necessary.
        // The rest of the code is already set up to be lazy about replacing bound vars,
        // and only when we actually have to normalize.
        let universes = if value.has_escaping_bound_vars() {
            let mut max_visitor =
                MaxEscapingBoundVarVisitor { outer_index: ty::INNERMOST, escaping: 0 };
            value.visit_with(&mut max_visitor);
            vec![None; max_visitor.escaping]
        } else {
            vec![]
        };

        if self.infcx.next_trait_solver() {
            match crate::solve::deeply_normalize_with_skipped_universes::<_, ScrubbedTraitError<'tcx>>(
                self, value, universes,
            ) {
                Ok(value) => {
                    return Ok(Normalized { value, obligations: PredicateObligations::new() });
                }
                Err(_errors) => {
                    return Err(NoSolution);
                }
            }
        }

        if !needs_normalization(self.infcx, &value) {
            return Ok(Normalized { value, obligations: PredicateObligations::new() });
        }

        let mut normalizer = QueryNormalizer {
            infcx: self.infcx,
            cause: self.cause,
            param_env: self.param_env,
            obligations: PredicateObligations::new(),
            cache: SsoHashMap::new(),
            anon_depth: 0,
            universes,
        };

        let result = value.try_fold_with(&mut normalizer);
        info!(
            "normalize::<{}>: result={:?} with {} obligations",
            std::any::type_name::<T>(),
            result,
            normalizer.obligations.len(),
        );
        debug!(
            "normalize::<{}>: obligations={:?}",
            std::any::type_name::<T>(),
            normalizer.obligations,
        );
        result.map(|value| Normalized { value, obligations: normalizer.obligations })
    }
}

// Visitor to find the maximum escaping bound var
struct MaxEscapingBoundVarVisitor {
    // The index which would count as escaping
    outer_index: ty::DebruijnIndex,
    escaping: usize,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for MaxEscapingBoundVarVisitor {
    fn visit_binder<T: TypeVisitable<TyCtxt<'tcx>>>(&mut self, t: &ty::Binder<'tcx, T>) {
        self.outer_index.shift_in(1);
        t.super_visit_with(self);
        self.outer_index.shift_out(1);
    }

    #[inline]
    fn visit_ty(&mut self, t: Ty<'tcx>) {
        if t.outer_exclusive_binder() > self.outer_index {
            self.escaping = self
                .escaping
                .max(t.outer_exclusive_binder().as_usize() - self.outer_index.as_usize());
        }
    }

    #[inline]
    fn visit_region(&mut self, r: ty::Region<'tcx>) {
        match r.kind() {
            ty::ReBound(debruijn, _) if debruijn > self.outer_index => {
                self.escaping =
                    self.escaping.max(debruijn.as_usize() - self.outer_index.as_usize());
            }
            _ => {}
        }
    }

    fn visit_const(&mut self, ct: ty::Const<'tcx>) {
        if ct.outer_exclusive_binder() > self.outer_index {
            self.escaping = self
                .escaping
                .max(ct.outer_exclusive_binder().as_usize() - self.outer_index.as_usize());
        }
    }
}

struct QueryNormalizer<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    cause: &'a ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    obligations: PredicateObligations<'tcx>,
    cache: SsoHashMap<Ty<'tcx>, Ty<'tcx>>,
    anon_depth: usize,
    universes: Vec<Option<ty::UniverseIndex>>,
}

impl<'a, 'tcx> FallibleTypeFolder<TyCtxt<'tcx>> for QueryNormalizer<'a, 'tcx> {
    type Error = NoSolution;

    fn cx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn try_fold_binder<T: TypeFoldable<TyCtxt<'tcx>>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> Result<ty::Binder<'tcx, T>, Self::Error> {
        self.universes.push(None);
        let t = t.try_super_fold_with(self);
        self.universes.pop();
        t
    }

    #[instrument(level = "debug", skip(self))]
    fn try_fold_ty(&mut self, ty: Ty<'tcx>) -> Result<Ty<'tcx>, Self::Error> {
        if !needs_normalization(self.infcx, &ty) {
            return Ok(ty);
        }

        if let Some(ty) = self.cache.get(&ty) {
            return Ok(*ty);
        }

        let (kind, data) = match *ty.kind() {
            ty::Alias(kind, data) => (kind, data),
            _ => {
                let res = ty.try_super_fold_with(self)?;
                self.cache.insert(ty, res);
                return Ok(res);
            }
        };

        // See note in `rustc_trait_selection::traits::project` about why we
        // wait to fold the args.
        let res = match kind {
            ty::Opaque => {
                // Only normalize `impl Trait` outside of type inference, usually in codegen.
                match self.infcx.typing_mode() {
                    TypingMode::Coherence
                    | TypingMode::Analysis { .. }
                    | TypingMode::Borrowck { .. }
                    | TypingMode::PostBorrowckAnalysis { .. } => ty.try_super_fold_with(self)?,

                    TypingMode::PostAnalysis => {
                        let args = data.args.try_fold_with(self)?;
                        let recursion_limit = self.cx().recursion_limit();

                        if !recursion_limit.value_within_limit(self.anon_depth) {
                            let guar = self
                                .infcx
                                .err_ctxt()
                                .build_overflow_error(
                                    OverflowCause::DeeplyNormalize(data.into()),
                                    self.cause.span,
                                    true,
                                )
                                .delay_as_bug();
                            return Ok(Ty::new_error(self.cx(), guar));
                        }

                        let generic_ty = self.cx().type_of(data.def_id);
                        let mut concrete_ty = generic_ty.instantiate(self.cx(), args);
                        self.anon_depth += 1;
                        if concrete_ty == ty {
                            concrete_ty = Ty::new_error_with_message(
                                self.cx(),
                                DUMMY_SP,
                                "recursive opaque type",
                            );
                        }
                        let folded_ty = ensure_sufficient_stack(|| self.try_fold_ty(concrete_ty));
                        self.anon_depth -= 1;
                        folded_ty?
                    }
                }
            }

            ty::Projection | ty::Inherent | ty::Free => {
                // See note in `rustc_trait_selection::traits::project`

                let infcx = self.infcx;
                let tcx = infcx.tcx;
                // Just an optimization: When we don't have escaping bound vars,
                // we don't need to replace them with placeholders.
                let (data, maps) = if data.has_escaping_bound_vars() {
                    let (data, mapped_regions, mapped_types, mapped_consts) =
                        BoundVarReplacer::replace_bound_vars(infcx, &mut self.universes, data);
                    (data, Some((mapped_regions, mapped_types, mapped_consts)))
                } else {
                    (data, None)
                };
                let data = data.try_fold_with(self)?;

                let mut orig_values = OriginalQueryValues::default();
                let c_data = infcx.canonicalize_query(self.param_env.and(data), &mut orig_values);
                debug!("QueryNormalizer: c_data = {:#?}", c_data);
                debug!("QueryNormalizer: orig_values = {:#?}", orig_values);
                let result = match kind {
                    ty::Projection => tcx.normalize_canonicalized_projection_ty(c_data),
                    ty::Free => tcx.normalize_canonicalized_free_alias(c_data),
                    ty::Inherent => tcx.normalize_canonicalized_inherent_projection_ty(c_data),
                    kind => unreachable!("did not expect {kind:?} due to match arm above"),
                }?;
                // We don't expect ambiguity.
                if !result.value.is_proven() {
                    // Rustdoc normalizes possibly not well-formed types, so only
                    // treat this as a bug if we're not in rustdoc.
                    if !tcx.sess.opts.actually_rustdoc {
                        tcx.dcx()
                            .delayed_bug(format!("unexpected ambiguity: {c_data:?} {result:?}"));
                    }
                    return Err(NoSolution);
                }
                let InferOk { value: result, obligations } = infcx
                    .instantiate_query_response_and_region_obligations(
                        self.cause,
                        self.param_env,
                        &orig_values,
                        result,
                    )?;
                debug!("QueryNormalizer: result = {:#?}", result);
                debug!("QueryNormalizer: obligations = {:#?}", obligations);
                self.obligations.extend(obligations);
                let res = if let Some((mapped_regions, mapped_types, mapped_consts)) = maps {
                    PlaceholderReplacer::replace_placeholders(
                        infcx,
                        mapped_regions,
                        mapped_types,
                        mapped_consts,
                        &self.universes,
                        result.normalized_ty,
                    )
                } else {
                    result.normalized_ty
                };
                // `tcx.normalize_canonicalized_projection_ty` may normalize to a type that
                // still has unevaluated consts, so keep normalizing here if that's the case.
                // Similarly, `tcx.normalize_canonicalized_free_alias` will only unwrap one layer
                // of type and we need to continue folding it to reveal the TAIT behind it.
                if res != ty
                    && (res.has_type_flags(ty::TypeFlags::HAS_CT_PROJECTION) || kind == ty::Free)
                {
                    res.try_fold_with(self)?
                } else {
                    res
                }
            }
        };

        self.cache.insert(ty, res);
        Ok(res)
    }

    fn try_fold_const(
        &mut self,
        constant: ty::Const<'tcx>,
    ) -> Result<ty::Const<'tcx>, Self::Error> {
        if !needs_normalization(self.infcx, &constant) {
            return Ok(constant);
        }

        let constant = crate::traits::with_replaced_escaping_bound_vars(
            self.infcx,
            &mut self.universes,
            constant,
            |constant| crate::traits::evaluate_const(&self.infcx, constant, self.param_env),
        );
        debug!(?constant, ?self.param_env);
        constant.try_super_fold_with(self)
    }

    #[inline]
    fn try_fold_predicate(
        &mut self,
        p: ty::Predicate<'tcx>,
    ) -> Result<ty::Predicate<'tcx>, Self::Error> {
        if p.allow_normalization() && needs_normalization(self.infcx, &p) {
            p.try_super_fold_with(self)
        } else {
            Ok(p)
        }
    }
}
