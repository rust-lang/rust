#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

use rustc_errors::{DiagnosticBuilder, ErrorGuaranteed};
use rustc_infer::infer::canonical::Canonical;
use rustc_infer::infer::error_reporting::nice_region_error::NiceRegionError;
use rustc_infer::infer::region_constraints::Constraint;
use rustc_infer::infer::region_constraints::RegionConstraintData;
use rustc_infer::infer::RegionVariableOrigin;
use rustc_infer::infer::{InferCtxt, RegionResolutionError, SubregionOrigin, TyCtxtInferExt as _};
use rustc_infer::traits::ObligationCause;
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::RegionVid;
use rustc_middle::ty::UniverseIndex;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc_span::Span;
use rustc_trait_selection::traits::query::type_op;
use rustc_trait_selection::traits::ObligationCtxt;
use rustc_traits::{type_op_ascribe_user_type_with_span, type_op_prove_predicate_with_cause};

use std::fmt;
use std::rc::Rc;

use crate::region_infer::values::RegionElement;
use crate::session_diagnostics::HigherRankedErrorCause;
use crate::session_diagnostics::HigherRankedLifetimeError;
use crate::session_diagnostics::HigherRankedSubtypeError;
use crate::MirBorrowckCtxt;

#[derive(Clone)]
pub(crate) struct UniverseInfo<'tcx>(UniverseInfoInner<'tcx>);

/// What operation a universe was created for.
#[derive(Clone)]
enum UniverseInfoInner<'tcx> {
    /// Relating two types which have binders.
    RelateTys { expected: Ty<'tcx>, found: Ty<'tcx> },
    /// Created from performing a `TypeOp`.
    TypeOp(Rc<dyn TypeOpInfo<'tcx> + 'tcx>),
    /// Any other reason.
    Other,
}

impl<'tcx> UniverseInfo<'tcx> {
    pub(crate) fn other() -> UniverseInfo<'tcx> {
        UniverseInfo(UniverseInfoInner::Other)
    }

    pub(crate) fn relate(expected: Ty<'tcx>, found: Ty<'tcx>) -> UniverseInfo<'tcx> {
        UniverseInfo(UniverseInfoInner::RelateTys { expected, found })
    }

    pub(crate) fn report_error(
        &self,
        mbcx: &mut MirBorrowckCtxt<'_, 'tcx>,
        placeholder: ty::PlaceholderRegion,
        error_element: RegionElement,
        cause: ObligationCause<'tcx>,
    ) {
        match self.0 {
            UniverseInfoInner::RelateTys { expected, found } => {
                let err = mbcx.infcx.err_ctxt().report_mismatched_types(
                    &cause,
                    expected,
                    found,
                    TypeError::RegionsPlaceholderMismatch,
                );
                mbcx.buffer_error(err);
            }
            UniverseInfoInner::TypeOp(ref type_op_info) => {
                type_op_info.report_error(mbcx, placeholder, error_element, cause);
            }
            UniverseInfoInner::Other => {
                // FIXME: This error message isn't great, but it doesn't show
                // up in the existing UI tests. Consider investigating this
                // some more.
                mbcx.buffer_error(
                    mbcx.infcx.tcx.sess.create_err(HigherRankedSubtypeError { span: cause.span }),
                );
            }
        }
    }
}

pub(crate) trait ToUniverseInfo<'tcx> {
    fn to_universe_info(self, base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx>;
}

impl<'tcx> ToUniverseInfo<'tcx> for crate::type_check::InstantiateOpaqueType<'tcx> {
    fn to_universe_info(self, base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        UniverseInfo(UniverseInfoInner::TypeOp(Rc::new(crate::type_check::InstantiateOpaqueType {
            base_universe: Some(base_universe),
            ..self
        })))
    }
}

impl<'tcx> ToUniverseInfo<'tcx>
    for Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::prove_predicate::ProvePredicate<'tcx>>>
{
    fn to_universe_info(self, base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        UniverseInfo(UniverseInfoInner::TypeOp(Rc::new(PredicateQuery {
            canonical_query: self,
            base_universe,
        })))
    }
}

impl<'tcx, T: Copy + fmt::Display + TypeFoldable<TyCtxt<'tcx>> + 'tcx> ToUniverseInfo<'tcx>
    for Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::Normalize<T>>>
{
    fn to_universe_info(self, base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        UniverseInfo(UniverseInfoInner::TypeOp(Rc::new(NormalizeQuery {
            canonical_query: self,
            base_universe,
        })))
    }
}

impl<'tcx> ToUniverseInfo<'tcx>
    for Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::AscribeUserType<'tcx>>>
{
    fn to_universe_info(self, base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        UniverseInfo(UniverseInfoInner::TypeOp(Rc::new(AscribeUserTypeQuery {
            canonical_query: self,
            base_universe,
        })))
    }
}

impl<'tcx, F> ToUniverseInfo<'tcx> for Canonical<'tcx, type_op::custom::CustomTypeOp<F>> {
    fn to_universe_info(self, _base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        // We can't rerun custom type ops.
        UniverseInfo::other()
    }
}

impl<'tcx> ToUniverseInfo<'tcx> for ! {
    fn to_universe_info(self, _base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        self
    }
}

#[allow(unused_lifetimes)]
trait TypeOpInfo<'tcx> {
    /// Returns an error to be reported if rerunning the type op fails to
    /// recover the error's cause.
    fn fallback_error(
        &self,
        tcx: TyCtxt<'tcx>,
        span: Span,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed>;

    fn base_universe(&self) -> ty::UniverseIndex;

    fn nice_error(
        &self,
        mbcx: &mut MirBorrowckCtxt<'_, 'tcx>,
        cause: ObligationCause<'tcx>,
        placeholder_region: ty::Region<'tcx>,
        error_region: Option<ty::Region<'tcx>>,
    ) -> Option<DiagnosticBuilder<'tcx, ErrorGuaranteed>>;

    #[instrument(level = "debug", skip(self, mbcx))]
    fn report_error(
        &self,
        mbcx: &mut MirBorrowckCtxt<'_, 'tcx>,
        placeholder: ty::PlaceholderRegion,
        error_element: RegionElement,
        cause: ObligationCause<'tcx>,
    ) {
        let tcx = mbcx.infcx.tcx;
        let base_universe = self.base_universe();
        debug!(?base_universe);

        let Some(adjusted_universe) =
            placeholder.universe.as_u32().checked_sub(base_universe.as_u32())
        else {
            mbcx.buffer_error(self.fallback_error(tcx, cause.span));
            return;
        };

        let placeholder_region = ty::Region::new_placeholder(
            tcx,
            ty::Placeholder { universe: adjusted_universe.into(), bound: placeholder.bound },
        );

        let error_region = if let RegionElement::PlaceholderRegion(error_placeholder) =
            error_element
        {
            let adjusted_universe =
                error_placeholder.universe.as_u32().checked_sub(base_universe.as_u32());
            adjusted_universe.map(|adjusted| {
                ty::Region::new_placeholder(
                    tcx,
                    ty::Placeholder { universe: adjusted.into(), bound: error_placeholder.bound },
                )
            })
        } else {
            None
        };

        debug!(?placeholder_region);

        let span = cause.span;
        let nice_error = self.nice_error(mbcx, cause, placeholder_region, error_region);

        if let Some(nice_error) = nice_error {
            mbcx.buffer_error(nice_error);
        } else {
            mbcx.buffer_error(self.fallback_error(tcx, span));
        }
    }
}

struct PredicateQuery<'tcx> {
    canonical_query:
        Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::prove_predicate::ProvePredicate<'tcx>>>,
    base_universe: ty::UniverseIndex,
}

impl<'tcx> TypeOpInfo<'tcx> for PredicateQuery<'tcx> {
    fn fallback_error(
        &self,
        tcx: TyCtxt<'tcx>,
        span: Span,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        tcx.sess.create_err(HigherRankedLifetimeError {
            cause: Some(HigherRankedErrorCause::CouldNotProve {
                predicate: self.canonical_query.value.value.predicate.to_string(),
            }),
            span,
        })
    }

    fn base_universe(&self) -> ty::UniverseIndex {
        self.base_universe
    }

    fn nice_error(
        &self,
        mbcx: &mut MirBorrowckCtxt<'_, 'tcx>,
        cause: ObligationCause<'tcx>,
        placeholder_region: ty::Region<'tcx>,
        error_region: Option<ty::Region<'tcx>>,
    ) -> Option<DiagnosticBuilder<'tcx, ErrorGuaranteed>> {
        let (infcx, key, _) =
            mbcx.infcx.tcx.infer_ctxt().build_with_canonical(cause.span, &self.canonical_query);
        let ocx = ObligationCtxt::new(&infcx);
        type_op_prove_predicate_with_cause(&ocx, key, cause);
        try_extract_error_from_fulfill_cx(&ocx, placeholder_region, error_region)
    }
}

struct NormalizeQuery<'tcx, T> {
    canonical_query: Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::Normalize<T>>>,
    base_universe: ty::UniverseIndex,
}

impl<'tcx, T> TypeOpInfo<'tcx> for NormalizeQuery<'tcx, T>
where
    T: Copy + fmt::Display + TypeFoldable<TyCtxt<'tcx>> + 'tcx,
{
    fn fallback_error(
        &self,
        tcx: TyCtxt<'tcx>,
        span: Span,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        tcx.sess.create_err(HigherRankedLifetimeError {
            cause: Some(HigherRankedErrorCause::CouldNotNormalize {
                value: self.canonical_query.value.value.value.to_string(),
            }),
            span,
        })
    }

    fn base_universe(&self) -> ty::UniverseIndex {
        self.base_universe
    }

    fn nice_error(
        &self,
        mbcx: &mut MirBorrowckCtxt<'_, 'tcx>,
        cause: ObligationCause<'tcx>,
        placeholder_region: ty::Region<'tcx>,
        error_region: Option<ty::Region<'tcx>>,
    ) -> Option<DiagnosticBuilder<'tcx, ErrorGuaranteed>> {
        let (infcx, key, _) =
            mbcx.infcx.tcx.infer_ctxt().build_with_canonical(cause.span, &self.canonical_query);
        let ocx = ObligationCtxt::new(&infcx);

        // FIXME(lqd): Unify and de-duplicate the following with the actual
        // `rustc_traits::type_op::type_op_normalize` query to allow the span we need in the
        // `ObligationCause`. The normalization results are currently different between
        // `QueryNormalizeExt::query_normalize` used in the query and `normalize` called below:
        // the former fails to normalize the `nll/relate_tys/impl-fn-ignore-binder-via-bottom.rs` test.
        // Check after #85499 lands to see if its fixes have erased this difference.
        let (param_env, value) = key.into_parts();
        let _ = ocx.normalize(&cause, param_env, value.value);

        try_extract_error_from_fulfill_cx(&ocx, placeholder_region, error_region)
    }
}

struct AscribeUserTypeQuery<'tcx> {
    canonical_query: Canonical<'tcx, ty::ParamEnvAnd<'tcx, type_op::AscribeUserType<'tcx>>>,
    base_universe: ty::UniverseIndex,
}

impl<'tcx> TypeOpInfo<'tcx> for AscribeUserTypeQuery<'tcx> {
    fn fallback_error(
        &self,
        tcx: TyCtxt<'tcx>,
        span: Span,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        // FIXME: This error message isn't great, but it doesn't show up in the existing UI tests,
        // and is only the fallback when the nice error fails. Consider improving this some more.
        tcx.sess.create_err(HigherRankedLifetimeError { cause: None, span })
    }

    fn base_universe(&self) -> ty::UniverseIndex {
        self.base_universe
    }

    fn nice_error(
        &self,
        mbcx: &mut MirBorrowckCtxt<'_, 'tcx>,
        cause: ObligationCause<'tcx>,
        placeholder_region: ty::Region<'tcx>,
        error_region: Option<ty::Region<'tcx>>,
    ) -> Option<DiagnosticBuilder<'tcx, ErrorGuaranteed>> {
        let (infcx, key, _) =
            mbcx.infcx.tcx.infer_ctxt().build_with_canonical(cause.span, &self.canonical_query);
        let ocx = ObligationCtxt::new(&infcx);
        type_op_ascribe_user_type_with_span(&ocx, key, Some(cause.span)).ok()?;
        try_extract_error_from_fulfill_cx(&ocx, placeholder_region, error_region)
    }
}

impl<'tcx> TypeOpInfo<'tcx> for crate::type_check::InstantiateOpaqueType<'tcx> {
    fn fallback_error(
        &self,
        tcx: TyCtxt<'tcx>,
        span: Span,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        // FIXME: This error message isn't great, but it doesn't show up in the existing UI tests,
        // and is only the fallback when the nice error fails. Consider improving this some more.
        tcx.sess.create_err(HigherRankedLifetimeError { cause: None, span })
    }

    fn base_universe(&self) -> ty::UniverseIndex {
        self.base_universe.unwrap()
    }

    fn nice_error(
        &self,
        mbcx: &mut MirBorrowckCtxt<'_, 'tcx>,
        _cause: ObligationCause<'tcx>,
        placeholder_region: ty::Region<'tcx>,
        error_region: Option<ty::Region<'tcx>>,
    ) -> Option<DiagnosticBuilder<'tcx, ErrorGuaranteed>> {
        try_extract_error_from_region_constraints(
            mbcx.infcx,
            placeholder_region,
            error_region,
            self.region_constraints.as_ref().unwrap(),
            // We're using the original `InferCtxt` that we
            // started MIR borrowchecking with, so the region
            // constraints have already been taken. Use the data from
            // our `mbcx` instead.
            |vid| mbcx.regioncx.var_infos[vid].origin,
            |vid| mbcx.regioncx.var_infos[vid].universe,
        )
    }
}

#[instrument(skip(ocx), level = "debug")]
fn try_extract_error_from_fulfill_cx<'tcx>(
    ocx: &ObligationCtxt<'_, 'tcx>,
    placeholder_region: ty::Region<'tcx>,
    error_region: Option<ty::Region<'tcx>>,
) -> Option<DiagnosticBuilder<'tcx, ErrorGuaranteed>> {
    // We generally shouldn't have errors here because the query was
    // already run, but there's no point using `delay_span_bug`
    // when we're going to emit an error here anyway.
    let _errors = ocx.select_all_or_error();
    let region_constraints = ocx.infcx.with_region_constraints(|r| r.clone());
    try_extract_error_from_region_constraints(
        ocx.infcx,
        placeholder_region,
        error_region,
        &region_constraints,
        |vid| ocx.infcx.region_var_origin(vid),
        |vid| ocx.infcx.universe_of_region(ty::Region::new_var(ocx.infcx.tcx, vid)),
    )
}

#[instrument(level = "debug", skip(infcx, region_var_origin, universe_of_region))]
fn try_extract_error_from_region_constraints<'tcx>(
    infcx: &InferCtxt<'tcx>,
    placeholder_region: ty::Region<'tcx>,
    error_region: Option<ty::Region<'tcx>>,
    region_constraints: &RegionConstraintData<'tcx>,
    mut region_var_origin: impl FnMut(RegionVid) -> RegionVariableOrigin,
    mut universe_of_region: impl FnMut(RegionVid) -> UniverseIndex,
) -> Option<DiagnosticBuilder<'tcx, ErrorGuaranteed>> {
    let (sub_region, cause) =
        region_constraints.constraints.iter().find_map(|(constraint, cause)| {
            match *constraint {
                Constraint::RegSubReg(sub, sup) if sup == placeholder_region && sup != sub => {
                    Some((sub, cause.clone()))
                }
                // FIXME: Should this check the universe of the var?
                Constraint::VarSubReg(vid, sup) if sup == placeholder_region => {
                    Some((ty::Region::new_var(infcx.tcx, vid), cause.clone()))
                }
                _ => None,
            }
        })?;

    debug!(?sub_region, "cause = {:#?}", cause);
    let error = match (error_region, *sub_region) {
        (Some(error_region), ty::ReVar(vid)) => RegionResolutionError::SubSupConflict(
            vid,
            region_var_origin(vid),
            cause.clone(),
            error_region,
            cause.clone(),
            placeholder_region,
            vec![],
        ),
        (Some(error_region), _) => {
            RegionResolutionError::ConcreteFailure(cause.clone(), error_region, placeholder_region)
        }
        // Note universe here is wrong...
        (None, ty::ReVar(vid)) => RegionResolutionError::UpperBoundUniverseConflict(
            vid,
            region_var_origin(vid),
            universe_of_region(vid),
            cause.clone(),
            placeholder_region,
        ),
        (None, _) => {
            RegionResolutionError::ConcreteFailure(cause.clone(), sub_region, placeholder_region)
        }
    };
    NiceRegionError::new(&infcx.err_ctxt(), error).try_report_from_nll().or_else(|| {
        if let SubregionOrigin::Subtype(trace) = cause {
            Some(
                infcx
                    .err_ctxt()
                    .report_and_explain_type_error(*trace, TypeError::RegionsPlaceholderMismatch),
            )
        } else {
            None
        }
    })
}
