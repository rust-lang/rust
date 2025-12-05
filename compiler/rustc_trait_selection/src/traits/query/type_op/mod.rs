use std::fmt;

use rustc_errors::ErrorGuaranteed;
use rustc_hir::def_id::LocalDefId;
use rustc_infer::traits::PredicateObligations;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::{ParamEnvAnd, TyCtxt, TypeFoldable};
use rustc_span::Span;

use crate::infer::canonical::{
    CanonicalQueryInput, CanonicalQueryResponse, Certainty, OriginalQueryValues,
    QueryRegionConstraints,
};
use crate::infer::{InferCtxt, InferOk};
use crate::traits::{ObligationCause, ObligationCtxt};

pub mod ascribe_user_type;
pub mod custom;
pub mod implied_outlives_bounds;
pub mod normalize;
pub mod outlives;
pub mod prove_predicate;

pub use rustc_middle::traits::query::type_op::*;

use self::custom::scrape_region_constraints;

/// "Type ops" are used in NLL to perform some particular action and
/// extract out the resulting region constraints (or an error if it
/// cannot be completed).
pub trait TypeOp<'tcx>: Sized + fmt::Debug {
    type Output: fmt::Debug;
    type ErrorInfo;

    /// Processes the operation and all resulting obligations,
    /// returning the final result along with any region constraints
    /// (they will be given over to the NLL region solver).
    fn fully_perform(
        self,
        infcx: &InferCtxt<'tcx>,
        root_def_id: LocalDefId,
        span: Span,
    ) -> Result<TypeOpOutput<'tcx, Self>, ErrorGuaranteed>;
}

/// The output from performing a type op
pub struct TypeOpOutput<'tcx, Op: TypeOp<'tcx>> {
    /// The output from the type op.
    pub output: Op::Output,
    /// Any region constraints from performing the type op.
    pub constraints: Option<&'tcx QueryRegionConstraints<'tcx>>,
    /// Used for error reporting to be able to rerun the query
    pub error_info: Option<Op::ErrorInfo>,
}

/// "Query type ops" are type ops that are implemented using a
/// [canonical query][c]. The `Self` type here contains the kernel of
/// information needed to do the operation -- `TypeOp` is actually
/// implemented for `ParamEnvAnd<Self>`, since we always need to bring
/// along a parameter environment as well. For query type-ops, we will
/// first canonicalize the key and then invoke the query on the tcx,
/// which produces the resulting query region constraints.
///
/// [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html
pub trait QueryTypeOp<'tcx>: fmt::Debug + Copy + TypeFoldable<TyCtxt<'tcx>> + 'tcx {
    type QueryResponse: TypeFoldable<TyCtxt<'tcx>>;

    /// Give query the option for a simple fast path that never
    /// actually hits the tcx cache lookup etc. Return `Some(r)` with
    /// a final result or `None` to do the full path.
    fn try_fast_path(
        tcx: TyCtxt<'tcx>,
        key: &ParamEnvAnd<'tcx, Self>,
    ) -> Option<Self::QueryResponse>;

    /// Performs the actual query with the canonicalized key -- the
    /// real work happens here. This method is not given an `infcx`
    /// because it shouldn't need one -- and if it had access to one,
    /// it might do things like invoke `sub_regions`, which would be
    /// bad, because it would create subregion relationships that are
    /// not captured in the return value.
    fn perform_query(
        tcx: TyCtxt<'tcx>,
        canonicalized: CanonicalQueryInput<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Result<CanonicalQueryResponse<'tcx, Self::QueryResponse>, NoSolution>;

    /// In the new trait solver, we already do caching in the solver itself,
    /// so there's no need to canonicalize and cache via the query system.
    /// Additionally, even if we were to canonicalize, we'd still need to
    /// make sure to feed it predefined opaque types and the defining anchor
    /// and that would require duplicating all of the tcx queries. Instead,
    /// just perform these ops locally.
    fn perform_locally_with_next_solver(
        ocx: &ObligationCtxt<'_, 'tcx>,
        key: ParamEnvAnd<'tcx, Self>,
        span: Span,
    ) -> Result<Self::QueryResponse, NoSolution>;

    fn fully_perform_into(
        query_key: ParamEnvAnd<'tcx, Self>,
        infcx: &InferCtxt<'tcx>,
        output_query_region_constraints: &mut QueryRegionConstraints<'tcx>,
        span: Span,
    ) -> Result<
        (
            Self::QueryResponse,
            Option<CanonicalQueryInput<'tcx, ParamEnvAnd<'tcx, Self>>>,
            PredicateObligations<'tcx>,
            Certainty,
        ),
        NoSolution,
    > {
        if let Some(result) = QueryTypeOp::try_fast_path(infcx.tcx, &query_key) {
            return Ok((result, None, PredicateObligations::new(), Certainty::Proven));
        }

        let mut canonical_var_values = OriginalQueryValues::default();
        let old_param_env = query_key.param_env;
        let canonical_self = infcx.canonicalize_query(query_key, &mut canonical_var_values);
        let canonical_result = Self::perform_query(infcx.tcx, canonical_self)?;

        let InferOk { value, obligations } = infcx
            .instantiate_nll_query_response_and_region_obligations(
                &ObligationCause::dummy_with_span(span),
                old_param_env,
                &canonical_var_values,
                canonical_result,
                output_query_region_constraints,
            )?;

        Ok((value, Some(canonical_self), obligations, canonical_result.value.certainty))
    }
}

impl<'tcx, Q> TypeOp<'tcx> for ParamEnvAnd<'tcx, Q>
where
    Q: QueryTypeOp<'tcx>,
{
    type Output = Q::QueryResponse;
    type ErrorInfo = CanonicalQueryInput<'tcx, ParamEnvAnd<'tcx, Q>>;

    fn fully_perform(
        self,
        infcx: &InferCtxt<'tcx>,
        root_def_id: LocalDefId,
        span: Span,
    ) -> Result<TypeOpOutput<'tcx, Self>, ErrorGuaranteed> {
        // In the new trait solver, query type ops are performed locally. This
        // is because query type ops currently use the old canonicalizer, and
        // that doesn't preserve things like opaques which have been registered
        // during MIR typeck. Even after the old canonicalizer is gone, it's
        // probably worthwhile just keeping this run-locally logic, since we
        // probably don't gain much from caching here given the new solver does
        // caching internally.
        if infcx.next_trait_solver() {
            return Ok(scrape_region_constraints(
                infcx,
                root_def_id,
                "query type op",
                span,
                |ocx| QueryTypeOp::perform_locally_with_next_solver(ocx, self, span),
            )?
            .0);
        }

        let mut error_info = None;
        let mut region_constraints = QueryRegionConstraints::default();

        // HACK(type_alias_impl_trait): When moving an opaque type to hidden type mapping from the query to the current inferctxt,
        // we sometimes end up with `Opaque<'a> = Opaque<'b>` instead of an actual hidden type. In that case we don't register a
        // hidden type but just equate the lifetimes. Thus we need to scrape the region constraints even though we're also manually
        // collecting region constraints via `region_constraints`.
        let (mut output, _) =
            scrape_region_constraints(infcx, root_def_id, "fully_perform", span, |ocx| {
                let (output, ei, obligations, _) =
                    Q::fully_perform_into(self, infcx, &mut region_constraints, span)?;
                error_info = ei;

                ocx.register_obligations(obligations);
                Ok(output)
            })?;
        output.error_info = error_info;
        if let Some(QueryRegionConstraints { outlives, assumptions }) = output.constraints {
            region_constraints.outlives.extend(outlives.iter().cloned());
            region_constraints.assumptions.extend(assumptions.iter().cloned());
        }
        output.constraints = if region_constraints.is_empty() {
            None
        } else {
            Some(infcx.tcx.arena.alloc(region_constraints))
        };
        Ok(output)
    }
}
