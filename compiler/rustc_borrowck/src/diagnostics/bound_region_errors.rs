use std::fmt;
use std::rc::Rc;

use rustc_errors::Diag;
use rustc_hir::def_id::LocalDefId;
use rustc_infer::infer::region_constraints::{Constraint, RegionConstraintData};
use rustc_infer::infer::{
    InferCtxt, RegionResolutionError, RegionVariableOrigin, SubregionOrigin, TyCtxtInferExt as _,
};
use rustc_infer::traits::ObligationCause;
use rustc_infer::traits::query::{
    CanonicalTypeOpAscribeUserTypeGoal, CanonicalTypeOpDeeplyNormalizeGoal,
    CanonicalTypeOpNormalizeGoal, CanonicalTypeOpProvePredicateGoal,
};
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::{
    self, RePlaceholder, Region, RegionVid, Ty, TyCtxt, TypeFoldable, UniverseIndex,
};
use rustc_span::Span;
use rustc_trait_selection::error_reporting::InferCtxtErrorExt;
use rustc_trait_selection::error_reporting::infer::nice_region_error::NiceRegionError;
use rustc_trait_selection::traits::ObligationCtxt;
use rustc_traits::{type_op_ascribe_user_type_with_span, type_op_prove_predicate_with_cause};
use tracing::{debug, instrument};

use crate::MirBorrowckCtxt;
use crate::region_infer::values::RegionElement;
use crate::session_diagnostics::{
    HigherRankedErrorCause, HigherRankedLifetimeError, HigherRankedSubtypeError,
};

/// What operation a universe was created for.
#[derive(Clone)]
pub(crate) enum UniverseInfo<'tcx> {
    /// Relating two types which have binders.
    RelateTys { expected: Ty<'tcx>, found: Ty<'tcx> },
    /// Created from performing a `TypeOp`.
    TypeOp(Rc<dyn TypeOpInfo<'tcx> + 'tcx>),
    /// Any other reason.
    Other,
}

impl<'tcx> UniverseInfo<'tcx> {
    pub(crate) fn other() -> UniverseInfo<'tcx> {
        UniverseInfo::Other
    }

    pub(crate) fn relate(expected: Ty<'tcx>, found: Ty<'tcx>) -> UniverseInfo<'tcx> {
        UniverseInfo::RelateTys { expected, found }
    }

    pub(crate) fn report_error(
        &self,
        mbcx: &mut MirBorrowckCtxt<'_, '_, 'tcx>,
        placeholder: ty::PlaceholderRegion,
        error_element: RegionElement,
        cause: ObligationCause<'tcx>,
    ) {
        match *self {
            UniverseInfo::RelateTys { expected, found } => {
                let err = mbcx.infcx.err_ctxt().report_mismatched_types(
                    &cause,
                    mbcx.infcx.param_env,
                    expected,
                    found,
                    TypeError::RegionsPlaceholderMismatch,
                );
                mbcx.buffer_error(err);
            }
            UniverseInfo::TypeOp(ref type_op_info) => {
                type_op_info.report_error(mbcx, placeholder, error_element, cause);
            }
            UniverseInfo::Other => {
                // FIXME: This error message isn't great, but it doesn't show
                // up in the existing UI tests. Consider investigating this
                // some more.
                mbcx.buffer_error(
                    mbcx.dcx().create_err(HigherRankedSubtypeError { span: cause.span }),
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
        UniverseInfo::TypeOp(Rc::new(crate::type_check::InstantiateOpaqueType {
            base_universe: Some(base_universe),
            ..self
        }))
    }
}

impl<'tcx> ToUniverseInfo<'tcx> for CanonicalTypeOpProvePredicateGoal<'tcx> {
    fn to_universe_info(self, base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        UniverseInfo::TypeOp(Rc::new(PredicateQuery { canonical_query: self, base_universe }))
    }
}

impl<'tcx, T: Copy + fmt::Display + TypeFoldable<TyCtxt<'tcx>> + 'tcx> ToUniverseInfo<'tcx>
    for CanonicalTypeOpNormalizeGoal<'tcx, T>
{
    fn to_universe_info(self, base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        UniverseInfo::TypeOp(Rc::new(NormalizeQuery { canonical_query: self, base_universe }))
    }
}

impl<'tcx, T: Copy + fmt::Display + TypeFoldable<TyCtxt<'tcx>> + 'tcx> ToUniverseInfo<'tcx>
    for CanonicalTypeOpDeeplyNormalizeGoal<'tcx, T>
{
    fn to_universe_info(self, base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        UniverseInfo::TypeOp(Rc::new(DeeplyNormalizeQuery { canonical_query: self, base_universe }))
    }
}

impl<'tcx> ToUniverseInfo<'tcx> for CanonicalTypeOpAscribeUserTypeGoal<'tcx> {
    fn to_universe_info(self, base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        UniverseInfo::TypeOp(Rc::new(AscribeUserTypeQuery { canonical_query: self, base_universe }))
    }
}

impl<'tcx> ToUniverseInfo<'tcx> for ! {
    fn to_universe_info(self, _base_universe: ty::UniverseIndex) -> UniverseInfo<'tcx> {
        self
    }
}

#[allow(unused_lifetimes)]
pub(crate) trait TypeOpInfo<'tcx> {
    /// Returns an error to be reported if rerunning the type op fails to
    /// recover the error's cause.
    fn fallback_error(&self, tcx: TyCtxt<'tcx>, span: Span) -> Diag<'tcx>;

    fn base_universe(&self) -> ty::UniverseIndex;

    fn nice_error<'infcx>(
        &self,
        mbcx: &mut MirBorrowckCtxt<'_, 'infcx, 'tcx>,
        cause: ObligationCause<'tcx>,
        placeholder_region: ty::Region<'tcx>,
        error_region: Option<ty::Region<'tcx>>,
    ) -> Option<Diag<'infcx>>;

    #[instrument(level = "debug", skip(self, mbcx))]
    fn report_error(
        &self,
        mbcx: &mut MirBorrowckCtxt<'_, '_, 'tcx>,
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

        debug!(?nice_error);

        if let Some(nice_error) = nice_error {
            mbcx.buffer_error(nice_error);
        } else {
            mbcx.buffer_error(self.fallback_error(tcx, span));
        }
    }
}

struct PredicateQuery<'tcx> {
    canonical_query: CanonicalTypeOpProvePredicateGoal<'tcx>,
    base_universe: ty::UniverseIndex,
}

impl<'tcx> TypeOpInfo<'tcx> for PredicateQuery<'tcx> {
    fn fallback_error(&self, tcx: TyCtxt<'tcx>, span: Span) -> Diag<'tcx> {
        tcx.dcx().create_err(HigherRankedLifetimeError {
            cause: Some(HigherRankedErrorCause::CouldNotProve {
                predicate: self.canonical_query.canonical.value.value.predicate.to_string(),
            }),
            span,
        })
    }

    fn base_universe(&self) -> ty::UniverseIndex {
        self.base_universe
    }

    fn nice_error<'infcx>(
        &self,
        mbcx: &mut MirBorrowckCtxt<'_, 'infcx, 'tcx>,
        cause: ObligationCause<'tcx>,
        placeholder_region: ty::Region<'tcx>,
        error_region: Option<ty::Region<'tcx>>,
    ) -> Option<Diag<'infcx>> {
        let (infcx, key, _) =
            mbcx.infcx.tcx.infer_ctxt().build_with_canonical(cause.span, &self.canonical_query);
        let ocx = ObligationCtxt::new(&infcx);
        type_op_prove_predicate_with_cause(&ocx, key, cause);
        let diag = try_extract_error_from_fulfill_cx(
            &ocx,
            mbcx.mir_def_id(),
            placeholder_region,
            error_region,
        )?
        .with_dcx(mbcx.dcx());
        Some(diag)
    }
}

struct NormalizeQuery<'tcx, T> {
    canonical_query: CanonicalTypeOpNormalizeGoal<'tcx, T>,
    base_universe: ty::UniverseIndex,
}

impl<'tcx, T> TypeOpInfo<'tcx> for NormalizeQuery<'tcx, T>
where
    T: Copy + fmt::Display + TypeFoldable<TyCtxt<'tcx>> + 'tcx,
{
    fn fallback_error(&self, tcx: TyCtxt<'tcx>, span: Span) -> Diag<'tcx> {
        tcx.dcx().create_err(HigherRankedLifetimeError {
            cause: Some(HigherRankedErrorCause::CouldNotNormalize {
                value: self.canonical_query.canonical.value.value.value.to_string(),
            }),
            span,
        })
    }

    fn base_universe(&self) -> ty::UniverseIndex {
        self.base_universe
    }

    fn nice_error<'infcx>(
        &self,
        mbcx: &mut MirBorrowckCtxt<'_, 'infcx, 'tcx>,
        cause: ObligationCause<'tcx>,
        placeholder_region: ty::Region<'tcx>,
        error_region: Option<ty::Region<'tcx>>,
    ) -> Option<Diag<'infcx>> {
        let (infcx, key, _) =
            mbcx.infcx.tcx.infer_ctxt().build_with_canonical(cause.span, &self.canonical_query);
        let ocx = ObligationCtxt::new(&infcx);

        // FIXME(lqd): Unify and de-duplicate the following with the actual
        // `rustc_traits::type_op::type_op_normalize` query to allow the span we need in the
        // `ObligationCause`. The normalization results are currently different between
        // `QueryNormalizeExt::query_normalize` used in the query and `normalize` called below:
        // the former fails to normalize the `nll/relate_tys/impl-fn-ignore-binder-via-bottom.rs`
        // test. Check after #85499 lands to see if its fixes have erased this difference.
        let (param_env, value) = key.into_parts();
        let _ = ocx.normalize(&cause, param_env, value.value);

        let diag = try_extract_error_from_fulfill_cx(
            &ocx,
            mbcx.mir_def_id(),
            placeholder_region,
            error_region,
        )?
        .with_dcx(mbcx.dcx());
        Some(diag)
    }
}

struct DeeplyNormalizeQuery<'tcx, T> {
    canonical_query: CanonicalTypeOpDeeplyNormalizeGoal<'tcx, T>,
    base_universe: ty::UniverseIndex,
}

impl<'tcx, T> TypeOpInfo<'tcx> for DeeplyNormalizeQuery<'tcx, T>
where
    T: Copy + fmt::Display + TypeFoldable<TyCtxt<'tcx>> + 'tcx,
{
    fn fallback_error(&self, tcx: TyCtxt<'tcx>, span: Span) -> Diag<'tcx> {
        tcx.dcx().create_err(HigherRankedLifetimeError {
            cause: Some(HigherRankedErrorCause::CouldNotNormalize {
                value: self.canonical_query.canonical.value.value.value.to_string(),
            }),
            span,
        })
    }

    fn base_universe(&self) -> ty::UniverseIndex {
        self.base_universe
    }

    fn nice_error<'infcx>(
        &self,
        mbcx: &mut MirBorrowckCtxt<'_, 'infcx, 'tcx>,
        cause: ObligationCause<'tcx>,
        placeholder_region: ty::Region<'tcx>,
        error_region: Option<ty::Region<'tcx>>,
    ) -> Option<Diag<'infcx>> {
        let (infcx, key, _) =
            mbcx.infcx.tcx.infer_ctxt().build_with_canonical(cause.span, &self.canonical_query);
        let ocx = ObligationCtxt::new(&infcx);

        let (param_env, value) = key.into_parts();
        let _ = ocx.deeply_normalize(&cause, param_env, value.value);

        let diag = try_extract_error_from_fulfill_cx(
            &ocx,
            mbcx.mir_def_id(),
            placeholder_region,
            error_region,
        )?
        .with_dcx(mbcx.dcx());
        Some(diag)
    }
}

struct AscribeUserTypeQuery<'tcx> {
    canonical_query: CanonicalTypeOpAscribeUserTypeGoal<'tcx>,
    base_universe: ty::UniverseIndex,
}

impl<'tcx> TypeOpInfo<'tcx> for AscribeUserTypeQuery<'tcx> {
    fn fallback_error(&self, tcx: TyCtxt<'tcx>, span: Span) -> Diag<'tcx> {
        // FIXME: This error message isn't great, but it doesn't show up in the existing UI tests,
        // and is only the fallback when the nice error fails. Consider improving this some more.
        tcx.dcx().create_err(HigherRankedLifetimeError { cause: None, span })
    }

    fn base_universe(&self) -> ty::UniverseIndex {
        self.base_universe
    }

    fn nice_error<'infcx>(
        &self,
        mbcx: &mut MirBorrowckCtxt<'_, 'infcx, 'tcx>,
        cause: ObligationCause<'tcx>,
        placeholder_region: ty::Region<'tcx>,
        error_region: Option<ty::Region<'tcx>>,
    ) -> Option<Diag<'infcx>> {
        let (infcx, key, _) =
            mbcx.infcx.tcx.infer_ctxt().build_with_canonical(cause.span, &self.canonical_query);
        let ocx = ObligationCtxt::new(&infcx);
        type_op_ascribe_user_type_with_span(&ocx, key, cause.span).ok()?;
        let diag = try_extract_error_from_fulfill_cx(
            &ocx,
            mbcx.mir_def_id(),
            placeholder_region,
            error_region,
        )?
        .with_dcx(mbcx.dcx());
        Some(diag)
    }
}

impl<'tcx> TypeOpInfo<'tcx> for crate::type_check::InstantiateOpaqueType<'tcx> {
    fn fallback_error(&self, tcx: TyCtxt<'tcx>, span: Span) -> Diag<'tcx> {
        // FIXME: This error message isn't great, but it doesn't show up in the existing UI tests,
        // and is only the fallback when the nice error fails. Consider improving this some more.
        tcx.dcx().create_err(HigherRankedLifetimeError { cause: None, span })
    }

    fn base_universe(&self) -> ty::UniverseIndex {
        self.base_universe.unwrap()
    }

    fn nice_error<'infcx>(
        &self,
        mbcx: &mut MirBorrowckCtxt<'_, 'infcx, 'tcx>,
        _cause: ObligationCause<'tcx>,
        placeholder_region: ty::Region<'tcx>,
        error_region: Option<ty::Region<'tcx>>,
    ) -> Option<Diag<'infcx>> {
        try_extract_error_from_region_constraints(
            mbcx.infcx,
            mbcx.mir_def_id(),
            placeholder_region,
            error_region,
            self.region_constraints.as_ref().unwrap(),
            // We're using the original `InferCtxt` that we
            // started MIR borrowchecking with, so the region
            // constraints have already been taken. Use the data from
            // our `mbcx` instead.
            |vid| RegionVariableOrigin::Nll(mbcx.regioncx.definitions[vid].origin),
            |vid| mbcx.regioncx.definitions[vid].universe,
        )
    }
}

#[instrument(skip(ocx), level = "debug")]
fn try_extract_error_from_fulfill_cx<'a, 'tcx>(
    ocx: &ObligationCtxt<'a, 'tcx>,
    generic_param_scope: LocalDefId,
    placeholder_region: ty::Region<'tcx>,
    error_region: Option<ty::Region<'tcx>>,
) -> Option<Diag<'a>> {
    // We generally shouldn't have errors here because the query was
    // already run, but there's no point using `span_delayed_bug`
    // when we're going to emit an error here anyway.
    let _errors = ocx.select_all_or_error();
    let region_constraints = ocx.infcx.with_region_constraints(|r| r.clone());
    try_extract_error_from_region_constraints(
        ocx.infcx,
        generic_param_scope,
        placeholder_region,
        error_region,
        &region_constraints,
        |vid| ocx.infcx.region_var_origin(vid),
        |vid| ocx.infcx.universe_of_region(ty::Region::new_var(ocx.infcx.tcx, vid)),
    )
}

#[instrument(level = "debug", skip(infcx, region_var_origin, universe_of_region))]
fn try_extract_error_from_region_constraints<'a, 'tcx>(
    infcx: &'a InferCtxt<'tcx>,
    generic_param_scope: LocalDefId,
    placeholder_region: ty::Region<'tcx>,
    error_region: Option<ty::Region<'tcx>>,
    region_constraints: &RegionConstraintData<'tcx>,
    mut region_var_origin: impl FnMut(RegionVid) -> RegionVariableOrigin,
    mut universe_of_region: impl FnMut(RegionVid) -> UniverseIndex,
) -> Option<Diag<'a>> {
    let placeholder_universe = match placeholder_region.kind() {
        ty::RePlaceholder(p) => p.universe,
        ty::ReVar(vid) => universe_of_region(vid),
        _ => ty::UniverseIndex::ROOT,
    };
    let matches =
        |a_region: Region<'tcx>, b_region: Region<'tcx>| match (a_region.kind(), b_region.kind()) {
            (RePlaceholder(a_p), RePlaceholder(b_p)) => a_p.bound == b_p.bound,
            _ => a_region == b_region,
        };
    let mut check =
        |constraint: &Constraint<'tcx>, cause: &SubregionOrigin<'tcx>, exact| match *constraint {
            Constraint::RegSubReg(sub, sup)
                if ((exact && sup == placeholder_region)
                    || (!exact && matches(sup, placeholder_region)))
                    && sup != sub =>
            {
                Some((sub, cause.clone()))
            }
            Constraint::VarSubReg(vid, sup)
                if (exact
                    && sup == placeholder_region
                    && !universe_of_region(vid).can_name(placeholder_universe))
                    || (!exact && matches(sup, placeholder_region)) =>
            {
                Some((ty::Region::new_var(infcx.tcx, vid), cause.clone()))
            }
            _ => None,
        };
    let mut info = region_constraints
        .constraints
        .iter()
        .find_map(|(constraint, cause)| check(constraint, cause, true));
    if info.is_none() {
        info = region_constraints
            .constraints
            .iter()
            .find_map(|(constraint, cause)| check(constraint, cause, false));
    }
    let (sub_region, cause) = info?;

    debug!(?sub_region, "cause = {:#?}", cause);
    let error = match (error_region, sub_region.kind()) {
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
    NiceRegionError::new(&infcx.err_ctxt(), generic_param_scope, error)
        .try_report_from_nll()
        .or_else(|| {
            if let SubregionOrigin::Subtype(trace) = cause {
                Some(infcx.err_ctxt().report_and_explain_type_error(
                    *trace,
                    infcx.tcx.param_env(generic_param_scope),
                    TypeError::RegionsPlaceholderMismatch,
                ))
            } else {
                None
            }
        })
}
