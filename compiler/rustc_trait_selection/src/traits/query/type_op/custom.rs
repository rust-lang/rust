use std::fmt;

use rustc_errors::ErrorGuaranteed;
use rustc_hir::def_id::LocalDefId;
use rustc_infer::infer::region_constraints::RegionConstraintData;
use rustc_infer::traits::{FromSolverError, ScrubbedTraitError};
use rustc_middle::traits::query::NoSolution;
use rustc_middle::ty::{TyCtxt, TypeFoldable};
use rustc_span::Span;
use tracing::info;

use crate::infer::InferCtxt;
use crate::infer::canonical::query_response;
use crate::solve::NextSolverError;
use crate::traits::query::type_op::TypeOpOutput;
use crate::traits::{FulfillmentError, ObligationCtxt, OldSolverError};

pub struct CustomTypeOp<F> {
    closure: F,
    description: &'static str,
}

impl<F> CustomTypeOp<F> {
    pub fn new<'tcx, R>(closure: F, description: &'static str) -> Self
    where
        F: FnOnce(&ObligationCtxt<'_, 'tcx>) -> Result<R, NoSolution>,
    {
        CustomTypeOp { closure, description }
    }
}

impl<'tcx, F, R> super::TypeOp<'tcx> for CustomTypeOp<F>
where
    F: FnOnce(&ObligationCtxt<'_, 'tcx>) -> Result<R, NoSolution>,
    R: fmt::Debug + TypeFoldable<TyCtxt<'tcx>>,
{
    type Output = R;
    /// We can't do any custom error reporting for `CustomTypeOp`, so
    /// we can use `!` to enforce that the implementation never provides it.
    type ErrorInfo = !;

    /// Processes the operation and all resulting obligations,
    /// returning the final result along with any region constraints
    /// (they will be given over to the NLL region solver).
    fn fully_perform(
        self,
        infcx: &InferCtxt<'tcx>,
        root_def_id: LocalDefId,
        span: Span,
    ) -> Result<TypeOpOutput<'tcx, Self>, ErrorGuaranteed> {
        if cfg!(debug_assertions) {
            info!("fully_perform({:?})", self);
        }

        Ok(scrape_region_constraints::<_, _, ScrubbedTraitError<'_>>(
            infcx,
            root_def_id,
            self.description,
            span,
            self.closure,
        )?
        .0)
    }
}

impl<F> fmt::Debug for CustomTypeOp<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.description.fmt(f)
    }
}

pub struct FallibleCustomTypeOp<F> {
    closure: F,
    description: &'static str,
}

impl<F> FallibleCustomTypeOp<F> {
    pub fn new<'tcx, R>(closure: F, description: &'static str) -> Self
    where
        F: FnOnce(&ObligationCtxt<'_, 'tcx, FulfillmentError<'tcx>>) -> Result<R, NoSolution>,
    {
        FallibleCustomTypeOp { closure, description }
    }
}

impl<'tcx, F, R> super::TypeOp<'tcx> for FallibleCustomTypeOp<F>
where
    F: FnOnce(&ObligationCtxt<'_, 'tcx, FulfillmentError<'tcx>>) -> Result<R, NoSolution>,
    R: fmt::Debug + TypeFoldable<TyCtxt<'tcx>>,
{
    type Output = R;
    type ErrorInfo = !;

    /// Processes the operation and all resulting obligations,
    /// returning the final result along with any region constraints
    /// (they will be given over to the NLL region solver).
    fn fully_perform(
        self,
        infcx: &InferCtxt<'tcx>,
        root_def_id: LocalDefId,
        span: Span,
    ) -> Result<TypeOpOutput<'tcx, Self>, ErrorGuaranteed> {
        if cfg!(debug_assertions) {
            info!("fully_perform({:?})", self);
        }

        Ok(scrape_region_constraints::<_, _, FulfillmentError<'tcx>>(
            infcx,
            root_def_id,
            self.description,
            span,
            self.closure,
        )?
        .0)
    }
}

impl<F> fmt::Debug for FallibleCustomTypeOp<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.description.fmt(f)
    }
}

/// Executes `op` and then scrapes out all the "old style" region
/// constraints that result, creating query-region-constraints.
pub fn scrape_region_constraints<'tcx, Op, R, E>(
    infcx: &InferCtxt<'tcx>,
    root_def_id: LocalDefId,
    name: &'static str,
    span: Span,
    op: impl FnOnce(&ObligationCtxt<'_, 'tcx, E>) -> Result<R, NoSolution>,
) -> Result<(TypeOpOutput<'tcx, Op>, RegionConstraintData<'tcx>), ErrorGuaranteed>
where
    R: TypeFoldable<TyCtxt<'tcx>>,
    Op: super::TypeOp<'tcx, Output = R>,
    E: FromSolverError<'tcx, NextSolverError<'tcx>> + FromSolverError<'tcx, OldSolverError<'tcx>>,
{
    // During NLL, we expect that nobody will register region
    // obligations **except** as part of a custom type op (and, at the
    // end of each custom type op, we scrape out the region
    // obligations that resulted). So this vector should be empty on
    // entry.
    let pre_obligations = infcx.take_registered_region_obligations();
    assert!(
        pre_obligations.is_empty(),
        "scrape_region_constraints: incoming region obligations = {pre_obligations:#?}",
    );
    let pre_assumptions = infcx.take_registered_region_assumptions();
    assert!(
        pre_assumptions.is_empty(),
        "scrape_region_constraints: incoming region assumptions = {pre_assumptions:#?}",
    );

    let value = infcx.commit_if_ok(|_| {
        let ocx = ObligationCtxt::<E>::new_in(infcx);
        op(&ocx)
            .and_then(|v| {
                let errors = ocx.evaluate_obligations_error_on_ambiguity();
                if errors.is_empty() {
                    Ok(v)
                } else {
                    E::try_report_errors(infcx, errors);
                    Err(NoSolution)
                }
            })
            .map_err(|_| {
                // FIXME: In this region-dependent context, `type_op` should only fail due to
                // region-dependent goals. Any other kind of failure indicates a bug and we
                // should ICE.
                //
                // In principle, all non-region errors are expected to be emitted before
                // borrowck. Such errors should taint the body and prevent borrowck from
                // running at all. However, this invariant does not currently hold.
                //
                // Consider:
                //
                // ```ignore (illustrative)
                // struct Foo<T>(T)
                // where
                //     T: Iterator,
                //     <T as Iterator>::Item: Default;
                //
                // fn foo<T>() -> Foo<T> {
                //     loop {}
                // }
                // ```
                //
                // Here, `fn foo` ought to error and taint the body before MIR build, since
                // `Foo<T>` is not well-formed. However, we currently avoid checking this
                // during HIR typeck to prevent duplicate diagnostics.
                //
                // If we ICE here on any non-region-dependent failure, we would trigger ICEs
                // too often for such cases. For now, we emit a delayed bug instead.
                match infcx.tcx.check_potentially_region_dependent_goals(root_def_id) {
                    Ok(_) => infcx
                        .dcx()
                        .span_delayed_bug(span, format!("error performing operation: {name}")),
                    Err(e) => e,
                }
            })
    })?;

    // Next trait solver performs operations locally, and normalize goals should resolve vars.
    let value = infcx.resolve_vars_if_possible(value);

    let region_obligations = infcx.take_registered_region_obligations();
    let region_assumptions = infcx.take_registered_region_assumptions();
    let region_constraint_data = infcx.take_and_reset_region_constraints();
    let region_constraints = query_response::make_query_region_constraints(
        region_obligations,
        &region_constraint_data,
        region_assumptions,
    );

    if region_constraints.is_empty() {
        Ok((
            TypeOpOutput { output: value, constraints: None, error_info: None },
            region_constraint_data,
        ))
    } else {
        Ok((
            TypeOpOutput {
                output: value,
                constraints: Some(infcx.tcx.arena.alloc(region_constraints)),
                error_info: None,
            },
            region_constraint_data,
        ))
    }
}
