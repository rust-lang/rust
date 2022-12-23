use rustc_infer::infer::canonical::CanonicalVarValues;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::query::NoSolution;
use rustc_middle::ty::Ty;
use rustc_span::DUMMY_SP;

use crate::solve::ExternalConstraints;

use super::{Certainty, QueryResult, Response};

/// Methods used inside of the canonical queries of the solver.
pub(super) trait InferCtxtExt<'tcx> {
    fn next_ty_infer(&self) -> Ty<'tcx>;

    fn make_canonical_response(
        &self,
        var_values: CanonicalVarValues<'tcx>,
        certainty: Certainty,
    ) -> QueryResult<'tcx>;
}

impl<'tcx> InferCtxtExt<'tcx> for InferCtxt<'tcx> {
    fn next_ty_infer(&self) -> Ty<'tcx> {
        self.next_ty_var(TypeVariableOrigin {
            kind: TypeVariableOriginKind::MiscVariable,
            span: DUMMY_SP,
        })
    }

    fn make_canonical_response(
        &self,
        var_values: CanonicalVarValues<'tcx>,
        certainty: Certainty,
    ) -> QueryResult<'tcx> {
        let external_constraints = take_external_constraints(self)?;

        Ok(self.canonicalize_response(Response { var_values, external_constraints, certainty }))
    }
}

#[instrument(level = "debug", skip(infcx), ret)]
fn take_external_constraints<'tcx>(
    infcx: &InferCtxt<'tcx>,
) -> Result<ExternalConstraints<'tcx>, NoSolution> {
    let region_obligations = infcx.take_registered_region_obligations();
    let opaque_types = infcx.take_opaque_types_for_query_response();
    Ok(ExternalConstraints {
        // FIXME: Now that's definitely wrong :)
        //
        // Should also do the leak check here I think
        regions: drop(region_obligations),
        opaque_types,
    })
}
