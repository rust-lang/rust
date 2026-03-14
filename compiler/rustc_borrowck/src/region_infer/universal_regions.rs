//! This module contains methods for checking universal
//! regions after region inference has executed.
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_middle::ty::{self, RegionVid};
use tracing::{debug, instrument};

use crate::constraints::OutlivesConstraintSet;
use crate::constraints::graph::NormalConstraintGraph;
use crate::consumers::PoloniusOutput;
use crate::diagnostics::{RegionErrorKind, RegionErrors};
use crate::handle_placeholders::RegionDefinitions;
use crate::region_infer::InferredRegions;
use crate::region_infer::constraint_search::ConstraintSearch;
use crate::region_infer::values::{LivenessValues, RegionElement};
use crate::{ClosureOutlivesRequirement, ClosureOutlivesSubject};

type MaybeOutlivesRequirements<'a, 'tcx> = Option<&'a mut Vec<ClosureOutlivesRequirement<'tcx>>>;

/// When we have an unmet lifetime constraint, we try to propagate it outward (e.g. to a closure
/// environment). If we can't, it is an error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RegionRelationCheckResult {
    Ok,
    Propagated,
    Error,
}

/// Context for checking universal region accesses.
pub(crate) struct UniversalRegionChecker<'a, 'tcx> {
    errors_buffer: &'a mut RegionErrors<'tcx>,
    region_definitions: &'a RegionDefinitions<'tcx>,
    constraints: &'a OutlivesConstraintSet<'tcx>,
    fr_static: RegionVid,
    values: &'a InferredRegions<'tcx>,
    constraint_graph: NormalConstraintGraph,
    liveness_constraints: &'a LivenessValues,
}

impl<'a, 'tcx> UniversalRegionChecker<'a, 'tcx> {
    /// Check universal region relations either after running Polonius,
    /// or after running regular borrow checking. See documentation for
    /// the methods below for an explanation of precisely what is checked.
    pub(super) fn check(
        mut self,
        polonius_output: Option<Box<PoloniusOutput>>,
        outlives_requirements: MaybeOutlivesRequirements<'a, 'tcx>,
    ) {
        // In Polonius mode, the errors about missing universal region relations are in the output
        // and need to be emitted or propagated. Otherwise, we need to check whether the
        // constraints were too strong, and if so, emit or propagate those errors.
        if let Some(polonius_output) = polonius_output {
            self.check_polonius_subset_errors(polonius_output.as_ref(), outlives_requirements);
        } else {
            self.check_universal_regions(outlives_requirements);
        }
    }

    pub(super) fn new(
        errors_buffer: &'a mut RegionErrors<'tcx>,
        region_definitions: &'a RegionDefinitions<'tcx>,
        constraints: &'a OutlivesConstraintSet<'tcx>,
        values: &'a InferredRegions<'tcx>,
        liveness_constraints: &'a LivenessValues,
    ) -> Self {
        Self {
            errors_buffer,
            region_definitions,
            constraints,
            fr_static: values.universal_region_relations.universal_regions.fr_static,
            constraint_graph: constraints.graph(region_definitions.len()),
            values,
            liveness_constraints,
        }
    }

    /// Once regions have been propagated, this method is used to see
    /// whether any of the constraints were too strong. In particular,
    /// we want to check for a case where a universally quantified
    /// region exceeded its bounds. Consider:
    /// ```compile_fail
    /// fn foo<'a, 'b>(x: &'a u32) -> &'b u32 { x }
    /// ```
    /// In this case, returning `x` requires `&'a u32 <: &'b u32`
    /// and hence we establish (transitively) a constraint that
    /// `'a: 'b`. The `propagate_constraints` code above will
    /// therefore add `end('a)` into the region for `'b` -- but we
    /// have no evidence that `'b` outlives `'a`, so we want to report
    /// an error.
    ///
    /// If `propagated_outlives_requirements` is `Some`, then we will
    /// push unsatisfied obligations into there. Otherwise, we'll
    /// report them as errors.
    fn check_universal_regions(
        &mut self,
        mut outlives_requirements: MaybeOutlivesRequirements<'a, 'tcx>,
    ) {
        for (fr, fr_definition) in self.region_definitions.iter_enumerated() {
            debug!(?fr, ?fr_definition);
            match fr_definition.origin {
                NllRegionVariableOrigin::FreeRegion => {
                    // Go through each of the universal regions `fr` and check that
                    // they did not grow too large, accumulating any requirements
                    // for our caller into the `outlives_requirements` vector.
                    self.check_universal_region(fr, &mut outlives_requirements);
                }

                NllRegionVariableOrigin::Placeholder(placeholder) => {
                    self.check_bound_universal_region(fr, placeholder);
                }

                NllRegionVariableOrigin::Existential { .. } => {
                    // nothing to check here
                }
            }
        }
    }

    /// Checks if Polonius has found any unexpected free region relations.
    ///
    /// In Polonius terms, a "subset error" (or "illegal subset relation error") is the equivalent
    /// of NLL's "checking if any region constraints were too strong": a placeholder origin `'a`
    /// was unexpectedly found to be a subset of another placeholder origin `'b`, and means in NLL
    /// terms that the "longer free region" `'a` outlived the "shorter free region" `'b`.
    ///
    /// More details can be found in this blog post by Niko:
    /// <https://smallcultfollowing.com/babysteps/blog/2019/01/17/polonius-and-region-errors/>
    ///
    /// In the canonical example
    /// ```compile_fail
    /// fn foo<'a, 'b>(x: &'a u32) -> &'b u32 { x }
    /// ```
    /// returning `x` requires `&'a u32 <: &'b u32` and hence we establish (transitively) a
    /// constraint that `'a: 'b`. It is an error that we have no evidence that this
    /// constraint holds.
    ///
    /// If `propagated_outlives_requirements` is `Some`, then we will
    /// push unsatisfied obligations into there. Otherwise, we'll
    /// report them as errors.
    fn check_polonius_subset_errors(
        &mut self,
        polonius_output: &PoloniusOutput,
        mut outlives_requirements: MaybeOutlivesRequirements<'a, 'tcx>,
    ) {
        debug!(
            "check_polonius_subset_errors: {} subset_errors",
            polonius_output.subset_errors.len()
        );

        // Similarly to `check_universal_regions`: a free region relation, which was not explicitly
        // declared ("known") was found by Polonius, so emit an error, or propagate the
        // requirements for our caller into the `propagated_outlives_requirements` vector.
        //
        // Polonius doesn't model regions ("origins") as CFG-subsets or durations, but the
        // `longer_fr` and `shorter_fr` terminology will still be used here, for consistency with
        // the rest of the NLL infrastructure. The "subset origin" is the "longer free region",
        // and the "superset origin" is the outlived "shorter free region".
        //
        // Note: Polonius will produce a subset error at every point where the unexpected
        // `longer_fr`'s "placeholder loan" is contained in the `shorter_fr`. This can be helpful
        // for diagnostics in the future, e.g. to point more precisely at the key locations
        // requiring this constraint to hold. However, the error and diagnostics code downstream
        // expects that these errors are not duplicated (and that they are in a certain order).
        // Otherwise, diagnostics messages such as the ones giving names like `'1` to elided or
        // anonymous lifetimes for example, could give these names differently, while others like
        // the outlives suggestions or the debug output from `#[rustc_regions]` would be
        // duplicated. The polonius subset errors are deduplicated here, while keeping the
        // CFG-location ordering.
        // We can iterate the HashMap here because the result is sorted afterwards.
        #[allow(rustc::potential_query_instability)]
        let mut subset_errors: Vec<_> = polonius_output
            .subset_errors
            .iter()
            .flat_map(|(_location, subset_errors)| subset_errors.iter())
            .collect();
        subset_errors.sort();
        subset_errors.dedup();

        for &(longer_fr, shorter_fr) in subset_errors.into_iter() {
            debug!(
                "check_polonius_subset_errors: subset_error longer_fr={:?},\
                 shorter_fr={:?}",
                longer_fr, shorter_fr
            );

            let propagated = self.try_propagate_universal_region_error(
                longer_fr.into(),
                shorter_fr.into(),
                &mut outlives_requirements,
            );
            if propagated == RegionRelationCheckResult::Error {
                self.errors_buffer.push(RegionErrorKind::RegionError {
                    longer_fr: longer_fr.into(),
                    shorter_fr: shorter_fr.into(),
                    fr_origin: NllRegionVariableOrigin::FreeRegion,
                    is_reported: true,
                });
            }
        }

        // Handle the placeholder errors as usual, until the chalk-rustc-polonius triumvirate has
        // a more complete picture on how to separate this responsibility.
        for (fr, fr_definition) in self.region_definitions.iter_enumerated() {
            match fr_definition.origin {
                NllRegionVariableOrigin::FreeRegion => {
                    // handled by polonius above
                }

                NllRegionVariableOrigin::Placeholder(placeholder) => {
                    self.check_bound_universal_region(fr, placeholder);
                }

                NllRegionVariableOrigin::Existential { .. } => {
                    // nothing to check here
                }
            }
        }
    }

    /// Checks the final value for the free region `fr` to see if it
    /// grew too large. In particular, examine what `end(X)` points
    /// wound up in `fr`'s final value; for each `end(X)` where `X !=
    /// fr`, we want to check that `fr: X`. If not, that's either an
    /// error, or something we have to propagate to our creator.
    ///
    /// Things that are to be propagated are accumulated into the
    /// `outlives_requirements` vector.
    #[instrument(skip(self), level = "debug")]
    fn check_universal_region(
        &mut self,
        longer_fr: RegionVid,
        mut outlives_requirements: &mut MaybeOutlivesRequirements<'a, 'tcx>,
    ) {
        // Because this free region must be in the ROOT universe, we
        // know it cannot contain any bound universes.
        assert!(self.values.max_nameable_universe(longer_fr).is_root());

        // Only check all of the relations for the main representative of each
        // SCC, otherwise just check that we outlive said representative. This
        // reduces the number of redundant relations propagated out of
        // closures.
        // Note that the representative will be a universal region if there is
        // one in this SCC, so we will always check the representative here.
        let representative = self.values.to_representative(longer_fr);
        if representative != longer_fr {
            if let RegionRelationCheckResult::Error = self.check_universal_region_relation(
                longer_fr,
                representative,
                &mut outlives_requirements,
            ) {
                self.errors_buffer.push(RegionErrorKind::RegionError {
                    longer_fr,
                    shorter_fr: representative,
                    fr_origin: NllRegionVariableOrigin::FreeRegion,
                    is_reported: true,
                });
            }
            return;
        }

        // Find every region `o` such that `fr: o`
        // (because `fr` includes `end(o)`).
        let mut error_reported = false;
        for shorter_fr in self.values.universal_regions_outlived_by(longer_fr) {
            if let RegionRelationCheckResult::Error = self.check_universal_region_relation(
                longer_fr,
                shorter_fr,
                &mut outlives_requirements,
            ) {
                // We only report the first region error. Subsequent errors are hidden so as
                // not to overwhelm the user, but we do record them so as to potentially print
                // better diagnostics elsewhere...
                self.errors_buffer.push(RegionErrorKind::RegionError {
                    longer_fr,
                    shorter_fr,
                    fr_origin: NllRegionVariableOrigin::FreeRegion,
                    is_reported: !error_reported,
                });

                error_reported = true;
            }
        }
    }

    fn check_bound_universal_region(
        &mut self,
        longer_fr: RegionVid,
        placeholder: ty::PlaceholderRegion<'tcx>,
    ) {
        debug!("check_bound_universal_region(fr={:?}, placeholder={:?})", longer_fr, placeholder,);

        let longer_fr_scc = self.values.scc(longer_fr);
        debug!("check_bound_universal_region: longer_fr_scc={:?}", longer_fr_scc,);

        // If we have some bound universal region `'a`, then the only
        // elements it can contain is itself -- we don't know anything
        // else about it!
        if let Some(error_element) = self
            .values
            .scc_values
            .elements_contained_in(longer_fr_scc)
            .find(|e| *e != RegionElement::PlaceholderRegion(placeholder))
        {
            let illegally_outlived_r = self.constraint_search().region_from_element(
                self.liveness_constraints,
                longer_fr,
                &error_element,
            );
            // Stop after the first error, it gets too noisy otherwise, and does not provide more information.
            self.errors_buffer.push(RegionErrorKind::PlaceholderOutlivesIllegalRegion {
                longer_fr,
                illegally_outlived_r,
            });
        } else {
            debug!("check_bound_universal_region: all bounds satisfied");
        }
    }

    /// Checks that we can prove that `longer_fr: shorter_fr`. If we can't we attempt to propagate
    /// the constraint outward (e.g. to a closure environment), but if that fails, there is an
    /// error.
    fn check_universal_region_relation(
        &mut self,
        longer_fr: RegionVid,
        shorter_fr: RegionVid,
        outlives_requirements: &mut MaybeOutlivesRequirements<'a, 'tcx>,
    ) -> RegionRelationCheckResult {
        // If it is known that `fr: o`, carry on.
        if self.values.universal_region_relations.outlives(longer_fr, shorter_fr) {
            RegionRelationCheckResult::Ok
        } else {
            // If we are not in a context where we can't propagate errors, or we
            // could not shrink `fr` to something smaller, then just report an
            // error.
            //
            // Note: in this case, we use the unapproximated regions to report the
            // error. This gives better error messages in some cases.
            self.try_propagate_universal_region_error(longer_fr, shorter_fr, outlives_requirements)
        }
    }

    /// Attempt to propagate a region error (e.g. `'a: 'b`) that is not met to a closure's
    /// creator. If we cannot, then the caller should report an error to the user.
    fn try_propagate_universal_region_error(
        &mut self,
        longer_fr: RegionVid,
        shorter_fr: RegionVid,
        outlives_requirements: &mut MaybeOutlivesRequirements<'a, 'tcx>,
    ) -> RegionRelationCheckResult {
        if let Some(propagated_outlives_requirements) = outlives_requirements {
            // Shrink `longer_fr` until we find some non-local regions.
            // We'll call them `longer_fr-` -- they are ever so slightly smaller than
            // `longer_fr`.
            let longer_fr_minus =
                self.values.universal_region_relations.non_local_lower_bounds(longer_fr);

            debug!("try_propagate_universal_region_error: fr_minus={:?}", longer_fr_minus);

            // If we don't find a any non-local regions, we should error out as there is nothing
            // to propagate.
            if longer_fr_minus.is_empty() {
                return RegionRelationCheckResult::Error;
            }

            let blame_constraint = self
                .constraint_search()
                .best_blame_constraint(longer_fr, NllRegionVariableOrigin::FreeRegion, shorter_fr)
                .0;

            // Grow `shorter_fr` until we find some non-local regions.
            // We will always find at least one: `'static`. We'll call
            // them `shorter_fr+` -- they're ever so slightly larger
            // than `shorter_fr`.
            let shorter_fr_plus =
                self.values.universal_region_relations.non_local_upper_bounds(shorter_fr);
            debug!("try_propagate_universal_region_error: shorter_fr_plus={:?}", shorter_fr_plus);

            // We then create constraints `longer_fr-: shorter_fr+` that may or may not
            // be propagated (see below).
            let mut constraints = vec![];
            for fr_minus in longer_fr_minus {
                for shorter_fr_plus in &shorter_fr_plus {
                    constraints.push((fr_minus, *shorter_fr_plus));
                }
            }

            // We only need to propagate at least one of the constraints for
            // soundness. However, we want to avoid arbitrary choices here
            // and currently don't support returning OR constraints.
            //
            // If any of the `shorter_fr+` regions are already outlived by `longer_fr-`,
            // we propagate only those.
            //
            // Consider this example (`'b: 'a` == `a -> b`), where we try to propagate `'d: 'a`:
            // a --> b --> d
            //  \
            //   \-> c
            // Here, `shorter_fr+` of `'a` == `['b, 'c]`.
            // Propagating `'d: 'b` is correct and should occur; `'d: 'c` is redundant because of
            // `'d: 'b` and could reject valid code.
            //
            // So we filter the constraints to regions already outlived by `longer_fr-`, but if
            // the filter yields an empty set, we fall back to the original one.
            let subset: Vec<_> = constraints
                .iter()
                .filter(|&&(fr_minus, shorter_fr_plus)| {
                    self.values.eval_outlives(fr_minus, shorter_fr_plus)
                })
                .copied()
                .collect();
            let propagated_constraints = if subset.is_empty() { constraints } else { subset };
            debug!(
                "try_propagate_universal_region_error: constraints={:?}",
                propagated_constraints
            );

            assert!(
                !propagated_constraints.is_empty(),
                "Expected at least one constraint to propagate here"
            );

            for (fr_minus, fr_plus) in propagated_constraints {
                // Push the constraint `long_fr-: shorter_fr+`
                propagated_outlives_requirements.push(ClosureOutlivesRequirement {
                    subject: ClosureOutlivesSubject::Region(fr_minus),
                    outlived_free_region: fr_plus,
                    blame_span: blame_constraint.cause.span,
                    category: blame_constraint.category,
                });
            }
            return RegionRelationCheckResult::Propagated;
        }

        RegionRelationCheckResult::Error
    }
    fn constraint_search(&'a self) -> ConstraintSearch<'a, 'tcx> {
        ConstraintSearch {
            definitions: self.region_definitions,
            fr_static: self.fr_static,
            constraint_graph: &self.constraint_graph,
            constraints: self.constraints,
        }
    }
}
