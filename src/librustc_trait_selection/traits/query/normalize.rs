//! Code for the 'normalization' query. This consists of a wrapper
//! which folds deeply, invoking the underlying
//! `normalize_projection_ty` query when it encounters projections.

use crate::infer::at::At;
use crate::infer::canonical::OriginalQueryValues;
use crate::infer::{InferCtxt, InferOk};
use crate::traits::Normalized;
use crate::traits::{Obligation, ObligationCause, PredicateObligation, Reveal};
use rustc::ty::fold::{TypeFoldable, TypeFolder};
use rustc::ty::subst::Subst;
use rustc::ty::{self, Ty, TyCtxt};

use super::NoSolution;

pub use rustc::traits::query::NormalizationResult;

impl<'cx, 'tcx> At<'cx, 'tcx> {
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
    pub fn normalize<T>(&self, value: &T) -> Result<Normalized<'tcx, T>, NoSolution>
    where
        T: TypeFoldable<'tcx>,
    {
        debug!(
            "normalize::<{}>(value={:?}, param_env={:?})",
            ::std::any::type_name::<T>(),
            value,
            self.param_env,
        );
        if !value.has_projections() {
            return Ok(Normalized { value: value.clone(), obligations: vec![] });
        }

        let mut normalizer = QueryNormalizer {
            infcx: self.infcx,
            cause: self.cause,
            param_env: self.param_env,
            obligations: vec![],
            error: false,
            anon_depth: 0,
        };

        let value1 = value.fold_with(&mut normalizer);
        if normalizer.error {
            Err(NoSolution)
        } else {
            Ok(Normalized { value: value1, obligations: normalizer.obligations })
        }
    }
}

struct QueryNormalizer<'cx, 'tcx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,
    cause: &'cx ObligationCause<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    obligations: Vec<PredicateObligation<'tcx>>,
    error: bool,
    anon_depth: usize,
}

impl<'cx, 'tcx> TypeFolder<'tcx> for QueryNormalizer<'cx, 'tcx> {
    fn tcx<'c>(&'c self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !ty.has_projections() {
            return ty;
        }

        let ty = ty.super_fold_with(self);
        match ty.kind {
            ty::Opaque(def_id, substs) if !substs.has_escaping_bound_vars() => {
                // (*)
                // Only normalize `impl Trait` after type-checking, usually in codegen.
                match self.param_env.reveal {
                    Reveal::UserFacing => ty,

                    Reveal::All => {
                        let recursion_limit = *self.tcx().sess.recursion_limit.get();
                        if self.anon_depth >= recursion_limit {
                            let obligation = Obligation::with_depth(
                                self.cause.clone(),
                                recursion_limit,
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
                        let folded_ty = self.fold_ty(concrete_ty);
                        self.anon_depth -= 1;
                        folded_ty
                    }
                }
            }

            ty::Projection(ref data) if !data.has_escaping_bound_vars() => {
                // (*)
                // (*) This is kind of hacky -- we need to be able to
                // handle normalization within binders because
                // otherwise we wind up a need to normalize when doing
                // trait matching (since you can have a trait
                // obligation like `for<'a> T::B : Fn(&'a int)`), but
                // we can't normalize with bound regions in scope. So
                // far now we just ignore binders but only normalize
                // if all bound regions are gone (and then we still
                // have to renormalize whenever we instantiate a
                // binder). It would be better to normalize in a
                // binding-aware fashion.

                let tcx = self.infcx.tcx;

                let mut orig_values = OriginalQueryValues::default();
                // HACK(matthewjasper) `'static` is special-cased in selection,
                // so we cannot canonicalize it.
                let c_data = self
                    .infcx
                    .canonicalize_hr_query_hack(&self.param_env.and(*data), &mut orig_values);
                debug!("QueryNormalizer: c_data = {:#?}", c_data);
                debug!("QueryNormalizer: orig_values = {:#?}", orig_values);
                match tcx.normalize_projection_ty(c_data) {
                    Ok(result) => {
                        // We don't expect ambiguity.
                        if result.is_ambiguous() {
                            self.error = true;
                            return ty;
                        }

                        match self.infcx.instantiate_query_response_and_region_obligations(
                            self.cause,
                            self.param_env,
                            &orig_values,
                            &result,
                        ) {
                            Ok(InferOk { value: result, obligations }) => {
                                debug!("QueryNormalizer: result = {:#?}", result);
                                debug!("QueryNormalizer: obligations = {:#?}", obligations);
                                self.obligations.extend(obligations);
                                return result.normalized_ty;
                            }

                            Err(_) => {
                                self.error = true;
                                return ty;
                            }
                        }
                    }

                    Err(NoSolution) => {
                        self.error = true;
                        ty
                    }
                }
            }

            _ => ty,
        }
    }

    fn fold_const(&mut self, constant: &'tcx ty::Const<'tcx>) -> &'tcx ty::Const<'tcx> {
        constant.eval(self.infcx.tcx, self.param_env)
    }
}
