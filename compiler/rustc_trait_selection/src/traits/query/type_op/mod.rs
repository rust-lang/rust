use crate::infer::canonical::{
    Canonicalized, CanonicalizedQueryResponse, OriginalQueryValues, QueryRegionConstraints,
};
use crate::infer::{InferCtxt, InferOk};
use crate::traits::query::Fallible;
use crate::traits::ObligationCause;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::{ParamEnvAnd, TyCtxt};
use std::fmt;
use std::rc::Rc;

pub mod ascribe_user_type;
pub mod custom;
pub mod eq;
pub mod implied_outlives_bounds;
pub mod normalize;
pub mod outlives;
pub mod prove_predicate;
use self::prove_predicate::ProvePredicate;
pub mod subtype;

pub use rustc_middle::traits::query::type_op::*;

/// "Type ops" are used in NLL to perform some particular action and
/// extract out the resulting region constraints (or an error if it
/// cannot be completed).
pub trait TypeOp<'tcx>: Sized + fmt::Debug {
    type Output;

    /// Processes the operation and all resulting obligations,
    /// returning the final result along with any region constraints
    /// (they will be given over to the NLL region solver).
    fn fully_perform(
        self,
        infcx: &InferCtxt<'_, 'tcx>,
    ) -> Fallible<(Self::Output, Option<Rc<QueryRegionConstraints<'tcx>>>)>;
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
pub trait QueryTypeOp<'tcx>: fmt::Debug + Sized + TypeFoldable<'tcx> + 'tcx {
    type QueryResponse: TypeFoldable<'tcx>;

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
        canonicalized: Canonicalized<'tcx, ParamEnvAnd<'tcx, Self>>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, Self::QueryResponse>>;

    fn fully_perform_into(
        query_key: ParamEnvAnd<'tcx, Self>,
        infcx: &InferCtxt<'_, 'tcx>,
        output_query_region_constraints: &mut QueryRegionConstraints<'tcx>,
    ) -> Fallible<Self::QueryResponse> {
        if let Some(result) = QueryTypeOp::try_fast_path(infcx.tcx, &query_key) {
            return Ok(result);
        }

        // FIXME(#33684) -- We need to use
        // `canonicalize_hr_query_hack` here because of things
        // like the subtype query, which go awry around
        // `'static` otherwise.
        let mut canonical_var_values = OriginalQueryValues::default();
        let old_param_env = query_key.param_env;
        let canonical_self = infcx.canonicalize_hr_query_hack(query_key, &mut canonical_var_values);
        let canonical_result = Self::perform_query(infcx.tcx, canonical_self)?;

        let InferOk { value, obligations } = infcx
            .instantiate_nll_query_response_and_region_obligations(
                &ObligationCause::dummy(),
                old_param_env,
                &canonical_var_values,
                canonical_result,
                output_query_region_constraints,
            )?;

        // Typically, instantiating NLL query results does not
        // create obligations. However, in some cases there
        // are unresolved type variables, and unify them *can*
        // create obligations. In that case, we have to go
        // fulfill them. We do this via a (recursive) query.
        for obligation in obligations {
            let () = ProvePredicate::fully_perform_into(
                obligation.param_env.and(ProvePredicate::new(obligation.predicate)),
                infcx,
                output_query_region_constraints,
            )?;
        }

        Ok(value)
    }
}

impl<'tcx, Q> TypeOp<'tcx> for ParamEnvAnd<'tcx, Q>
where
    Q: QueryTypeOp<'tcx>,
{
    type Output = Q::QueryResponse;

    fn fully_perform(
        self,
        infcx: &InferCtxt<'_, 'tcx>,
    ) -> Fallible<(Self::Output, Option<Rc<QueryRegionConstraints<'tcx>>>)> {
        let mut region_constraints = QueryRegionConstraints::default();
        let r = Q::fully_perform_into(self, infcx, &mut region_constraints)?;

        // Promote the final query-region-constraints into a
        // (optional) ref-counted vector:
        let opt_qrc =
            if region_constraints.is_empty() { None } else { Some(Rc::new(region_constraints)) };

        Ok((r, opt_qrc))
    }
}
