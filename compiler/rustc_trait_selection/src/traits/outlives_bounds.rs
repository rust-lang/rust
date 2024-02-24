use crate::infer::InferCtxt;
use crate::traits::{ObligationCause, ObligationCtxt};
use rustc_data_structures::fx::FxIndexSet;
use rustc_infer::infer::resolve::OpportunisticRegionResolver;
use rustc_infer::infer::InferOk;
use rustc_middle::infer::canonical::{OriginalQueryValues, QueryRegionConstraints};
use rustc_middle::ty::{self, ParamEnv, Ty, TypeFolder, TypeVisitableExt};
use rustc_span::def_id::LocalDefId;

pub use rustc_middle::traits::query::OutlivesBound;

pub type BoundsCompat<'a, 'tcx: 'a> = impl Iterator<Item = OutlivesBound<'tcx>> + 'a;
pub type Bounds<'a, 'tcx: 'a> = impl Iterator<Item = OutlivesBound<'tcx>> + 'a;

/// Implied bounds are region relationships that we deduce
/// automatically. The idea is that (e.g.) a caller must check that a
/// function's argument types are well-formed immediately before
/// calling that fn, and hence the *callee* can assume that its
/// argument types are well-formed. This may imply certain relationships
/// between generic parameters. For example:
/// ```
/// fn foo<T>(x: &T) {}
/// ```
/// can only be called with a `'a` and `T` such that `&'a T` is WF.
/// For `&'a T` to be WF, `T: 'a` must hold. So we can assume `T: 'a`.
///
/// # Parameters
///
/// - `param_env`, the where-clauses in scope
/// - `body_id`, the body-id to use when normalizing assoc types.
///   Note that this may cause outlives obligations to be injected
///   into the inference context with this body-id.
/// - `ty`, the type that we are supposed to assume is WF.
#[instrument(level = "debug", skip(infcx, param_env, body_id), ret)]
fn implied_outlives_bounds<'a, 'tcx>(
    infcx: &'a InferCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    body_id: LocalDefId,
    ty: Ty<'tcx>,
    compat: bool,
) -> Vec<OutlivesBound<'tcx>> {
    let ty = infcx.resolve_vars_if_possible(ty);
    let ty = OpportunisticRegionResolver::new(infcx).fold_ty(ty);

    // We do not expect existential variables in implied bounds.
    // We may however encounter unconstrained lifetime variables
    // in very rare cases.
    //
    // See `ui/implied-bounds/implied-bounds-unconstrained-2.rs` for
    // an example.
    assert!(!ty.has_non_region_infer());

    let mut canonical_var_values = OriginalQueryValues::default();
    let canonical_ty = infcx.canonicalize_query(param_env.and(ty), &mut canonical_var_values);
    let implied_bounds_result = if compat {
        infcx.tcx.implied_outlives_bounds_compat(canonical_ty)
    } else {
        infcx.tcx.implied_outlives_bounds(canonical_ty)
    };
    let Ok(canonical_result) = implied_bounds_result else {
        return vec![];
    };

    let mut constraints = QueryRegionConstraints::default();
    let span = infcx.tcx.def_span(body_id);
    let Ok(InferOk { value: mut bounds, obligations }) = infcx
        .instantiate_nll_query_response_and_region_obligations(
            &ObligationCause::dummy_with_span(span),
            param_env,
            &canonical_var_values,
            canonical_result,
            &mut constraints,
        )
    else {
        return vec![];
    };
    assert_eq!(&obligations, &[]);

    // Because of #109628, we may have unexpected placeholders. Ignore them!
    // FIXME(#109628): panic in this case once the issue is fixed.
    bounds.retain(|bound| !bound.has_placeholders());

    if !constraints.is_empty() {
        debug!(?constraints);
        if !constraints.member_constraints.is_empty() {
            span_bug!(span, "{:#?}", constraints.member_constraints);
        }

        // Instantiation may have produced new inference variables and constraints on those
        // variables. Process these constraints.
        let ocx = ObligationCtxt::new(infcx);
        let cause = ObligationCause::misc(span, body_id);
        for &constraint in &constraints.outlives {
            ocx.register_obligation(infcx.query_outlives_constraint_to_obligation(
                constraint,
                cause.clone(),
                param_env,
            ));
        }

        let errors = ocx.select_all_or_error();
        if !errors.is_empty() {
            infcx.dcx().span_bug(
                span,
                "implied_outlives_bounds failed to solve obligations from instantiation",
            );
        }
    };

    bounds
}

#[extension(pub trait InferCtxtExt<'a, 'tcx>)]
impl<'a, 'tcx: 'a> InferCtxt<'tcx> {
    /// Do *NOT* call this directly.
    fn implied_bounds_tys_compat(
        &'a self,
        param_env: ParamEnv<'tcx>,
        body_id: LocalDefId,
        tys: &'a FxIndexSet<Ty<'tcx>>,
        compat: bool,
    ) -> BoundsCompat<'a, 'tcx> {
        tys.iter()
            .flat_map(move |ty| implied_outlives_bounds(self, param_env, body_id, *ty, compat))
    }

    /// If `-Z no-implied-bounds-compat` is set, calls `implied_bounds_tys_compat`
    /// with `compat` set to `true`, otherwise `false`.
    fn implied_bounds_tys(
        &'a self,
        param_env: ParamEnv<'tcx>,
        body_id: LocalDefId,
        tys: &'a FxIndexSet<Ty<'tcx>>,
    ) -> Bounds<'a, 'tcx> {
        tys.iter().flat_map(move |ty| {
            implied_outlives_bounds(
                self,
                param_env,
                body_id,
                *ty,
                !self.tcx.sess.opts.unstable_opts.no_implied_bounds_compat,
            )
        })
    }
}
