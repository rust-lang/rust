//! Code for the 'normalization' query. This consists of a wrapper
//! which folds deeply, invoking the underlying
//! `normalize_projection_ty` query when it encounters projections.

use crate::infer::at::At;
use crate::infer::canonical::OriginalQueryValues;
use crate::infer::{InferCtxt, InferOk};
use crate::traits::error_reporting::InferCtxtExt;
use crate::traits::project::needs_normalization;
use crate::traits::{Obligation, ObligationCause, PredicateObligation, Reveal};
use rustc_data_structures::sso::SsoHashMap;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_infer::traits::Normalized;
use rustc_middle::mir;
use rustc_middle::ty::fold::{TypeFoldable, TypeFolder};
use rustc_middle::ty::subst::Subst;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitor};

use std::ops::ControlFlow;

use super::NoSolution;

pub use rustc_middle::traits::query::NormalizationResult;

pub trait AtExt<'tcx> {
    fn normalize<T>(&self, value: T) -> Result<Normalized<'tcx, T>, NoSolution>
    where
        T: TypeFoldable<'tcx>;
}

impl<'cx, 'tcx> AtExt<'tcx> for At<'cx, 'tcx> {
    /// Normalize `value` in the context of the inference context,
    /// yielding a resulting type, or an error if `value` cannot be
    /// normalized. If you don't care about regions, you should prefer
    /// `normalize_erasing_regions`, which is more efficient.
    ///
    /// If the normalization succeeds and is unambiguous, returns back
    /// the normalized value along with various outlives relations (in
    /// the form of obligations that must be discharged).
    ///
    /// N.B., this will *eventually* be the main means of
    /// normalizing, but for now should be used only when we actually
    /// know that normalization will succeed, since error reporting
    /// and other details are still "under development".
    fn normalize<T>(&self, value: T) -> Result<Normalized<'tcx, T>, NoSolution>
    where
        T: TypeFoldable<'tcx>,
    {
        debug!(
            "normalize::<{}>(value={:?}, param_env={:?})",
            std::any::type_name::<T>(),
            value,
            self.param_env,
        );
        if !needs_normalization(&value, self.param_env.reveal()) {
            return Ok(Normalized { value, obligations: vec![] });
        }

        let mut normalizer = QueryNormalizer {
            infcx: self.infcx,
            cause: self.cause,
            param_env: self.param_env,
            obligations: vec![],
            error: false,
            cache: SsoHashMap::new(),
            anon_depth: 0,
            universes: vec![],
        };

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
        if value.has_escaping_bound_vars() {
            let mut max_visitor = MaxEscapingBoundVarVisitor {
                tcx: self.infcx.tcx,
                outer_index: ty::INNERMOST,
                escaping: 0,
            };
            value.visit_with(&mut max_visitor);
            if max_visitor.escaping > 0 {
                normalizer.universes.extend((0..max_visitor.escaping).map(|_| None));
            }
        }
        let result = value.fold_with(&mut normalizer);
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
        if normalizer.error {
            Err(NoSolution)
        } else {
            Ok(Normalized { value: result, obligations: normalizer.obligations })
        }
    }
}

/// Visitor to find the maximum escaping bound var
struct MaxEscapingBoundVarVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    // The index which would count as escaping
    outer_index: ty::DebruijnIndex,
    escaping: usize,
}

impl<'tcx> TypeVisitor<'tcx> for MaxEscapingBoundVarVisitor<'tcx> {
    fn tcx_for_anon_const_substs(&self) -> Option<TyCtxt<'tcx>> {
        Some(self.tcx)
    }

    fn visit_binder<T: TypeFoldable<'tcx>>(
        &mut self,
        t: &ty::Binder<'tcx, T>,
    ) -> ControlFlow<Self::BreakTy> {
        self.outer_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.outer_index.shift_out(1);
        result
    }

    #[inline]
    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        if t.outer_exclusive_binder() > self.outer_index {
            self.escaping = self
                .escaping
                .max(t.outer_exclusive_binder().as_usize() - self.outer_index.as_usize());
        }
        ControlFlow::CONTINUE
    }

    #[inline]
    fn visit_region(&mut self, r: ty::Region<'tcx>) -> ControlFlow<Self::BreakTy> {
        match *r {
            ty::ReLateBound(debruijn, _) if debruijn > self.outer_index => {
                self.escaping =
                    self.escaping.max(debruijn.as_usize() - self.outer_index.as_usize());
            }
            _ => {}
        }
        ControlFlow::CONTINUE
    }

    fn visit_const(&mut self, ct: &'tcx ty::Const<'tcx>) -> ControlFlow<Self::BreakTy> {
        match ct.val {
            ty::ConstKind::Bound(debruijn, _) if debruijn >= self.outer_index => {
                self.escaping =
                    self.escaping.max(debruijn.as_usize() - self.outer_index.as_usize());
                ControlFlow::CONTINUE
            }
            _ => ct.super_visit_with(self),
        }
    }
}

struct QueryNormalizer<'cx, 'tcx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,
    cause: &'cx ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    obligations: Vec<PredicateObligation<'tcx>>,
    cache: SsoHashMap<Ty<'tcx>, Ty<'tcx>>,
    error: bool,
    anon_depth: usize,
    universes: Vec<Option<ty::UniverseIndex>>,
}

impl<'cx, 'tcx> TypeFolder<'tcx> for QueryNormalizer<'cx, 'tcx> {
    fn tcx<'c>(&'c self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_binder<T: TypeFoldable<'tcx>>(
        &mut self,
        t: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        self.universes.push(None);
        let t = t.super_fold_with(self);
        self.universes.pop();
        t
    }

    #[instrument(level = "debug", skip(self))]
    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !needs_normalization(&ty, self.param_env.reveal()) {
            return ty;
        }

        if let Some(ty) = self.cache.get(&ty) {
            return ty;
        }

        // See note in `rustc_trait_selection::traits::project` about why we
        // wait to fold the substs.

        // Wrap this in a closure so we don't accidentally return from the outer function
        let res = (|| match *ty.kind() {
            // This is really important. While we *can* handle this, this has
            // severe performance implications for large opaque types with
            // late-bound regions. See `issue-88862` benchmark.
            ty::Opaque(def_id, substs) if !substs.has_escaping_bound_vars() => {
                // Only normalize `impl Trait` after type-checking, usually in codegen.
                match self.param_env.reveal() {
                    Reveal::UserFacing => ty.super_fold_with(self),

                    Reveal::All => {
                        let substs = substs.super_fold_with(self);
                        let recursion_limit = self.tcx().recursion_limit();
                        if !recursion_limit.value_within_limit(self.anon_depth) {
                            let obligation = Obligation::with_depth(
                                self.cause.clone(),
                                recursion_limit.0,
                                self.param_env,
                                ty,
                            );
                            self.infcx.report_overflow_error(&obligation, true);
                        }

                        let generic_ty = self.tcx().type_of(def_id);
                        let concrete_ty = generic_ty.subst(self.tcx(), substs);
                        self.anon_depth += 1;
                        if concrete_ty == ty {
                            bug!(
                                "infinite recursion generic_ty: {:#?}, substs: {:#?}, \
                                 concrete_ty: {:#?}, ty: {:#?}",
                                generic_ty,
                                substs,
                                concrete_ty,
                                ty
                            );
                        }
                        let folded_ty = ensure_sufficient_stack(|| self.fold_ty(concrete_ty));
                        self.anon_depth -= 1;
                        folded_ty
                    }
                }
            }

            ty::Projection(data) if !data.has_escaping_bound_vars() => {
                // This branch is just an optimization: when we don't have escaping bound vars,
                // we don't need to replace them with placeholders (see branch below).

                let tcx = self.infcx.tcx;
                let data = data.super_fold_with(self);

                let mut orig_values = OriginalQueryValues::default();
                // HACK(matthewjasper) `'static` is special-cased in selection,
                // so we cannot canonicalize it.
                let c_data = self
                    .infcx
                    .canonicalize_query_keep_static(self.param_env.and(data), &mut orig_values);
                debug!("QueryNormalizer: c_data = {:#?}", c_data);
                debug!("QueryNormalizer: orig_values = {:#?}", orig_values);
                match tcx.normalize_projection_ty(c_data) {
                    Ok(result) => {
                        // We don't expect ambiguity.
                        if result.is_ambiguous() {
                            self.error = true;
                            return ty.super_fold_with(self);
                        }

                        match self.infcx.instantiate_query_response_and_region_obligations(
                            self.cause,
                            self.param_env,
                            &orig_values,
                            result,
                        ) {
                            Ok(InferOk { value: result, obligations }) => {
                                debug!("QueryNormalizer: result = {:#?}", result);
                                debug!("QueryNormalizer: obligations = {:#?}", obligations);
                                self.obligations.extend(obligations);
                                result.normalized_ty
                            }

                            Err(_) => {
                                self.error = true;
                                ty.super_fold_with(self)
                            }
                        }
                    }

                    Err(NoSolution) => {
                        self.error = true;
                        ty.super_fold_with(self)
                    }
                }
            }

            ty::Projection(data) => {
                // See note in `rustc_trait_selection::traits::project`

                let tcx = self.infcx.tcx;
                let infcx = self.infcx;
                let (data, mapped_regions, mapped_types, mapped_consts) =
                    crate::traits::project::BoundVarReplacer::replace_bound_vars(
                        infcx,
                        &mut self.universes,
                        data,
                    );
                let data = data.super_fold_with(self);

                let mut orig_values = OriginalQueryValues::default();
                // HACK(matthewjasper) `'static` is special-cased in selection,
                // so we cannot canonicalize it.
                let c_data = self
                    .infcx
                    .canonicalize_query_keep_static(self.param_env.and(data), &mut orig_values);
                debug!("QueryNormalizer: c_data = {:#?}", c_data);
                debug!("QueryNormalizer: orig_values = {:#?}", orig_values);
                match tcx.normalize_projection_ty(c_data) {
                    Ok(result) => {
                        // We don't expect ambiguity.
                        if result.is_ambiguous() {
                            self.error = true;
                            return ty.super_fold_with(self);
                        }
                        match self.infcx.instantiate_query_response_and_region_obligations(
                            self.cause,
                            self.param_env,
                            &orig_values,
                            result,
                        ) {
                            Ok(InferOk { value: result, obligations }) => {
                                debug!("QueryNormalizer: result = {:#?}", result);
                                debug!("QueryNormalizer: obligations = {:#?}", obligations);
                                self.obligations.extend(obligations);
                                crate::traits::project::PlaceholderReplacer::replace_placeholders(
                                    infcx,
                                    mapped_regions,
                                    mapped_types,
                                    mapped_consts,
                                    &self.universes,
                                    result.normalized_ty,
                                )
                            }
                            Err(_) => {
                                self.error = true;
                                ty.super_fold_with(self)
                            }
                        }
                    }
                    Err(NoSolution) => {
                        self.error = true;
                        ty.super_fold_with(self)
                    }
                }
            }

            _ => ty.super_fold_with(self),
        })();
        self.cache.insert(ty, res);
        res
    }

    fn fold_const(&mut self, constant: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        let constant = constant.super_fold_with(self);
        constant.eval(self.infcx.tcx, self.param_env)
    }

    fn fold_mir_const(&mut self, constant: mir::ConstantKind<'tcx>) -> mir::ConstantKind<'tcx> {
        constant.super_fold_with(self)
    }
}
