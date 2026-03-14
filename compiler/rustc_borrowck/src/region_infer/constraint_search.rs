//! This module contains code for searching the region
//! constraint graph, usually to find a good reason for why
//! one region is live or outlives another.

use std::collections::VecDeque;

use rustc_hir::def_id::CRATE_DEF_ID;
use rustc_index::IndexVec;
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_infer::traits::{ObligationCause, ObligationCauseCode};
use rustc_middle::bug;
use rustc_middle::mir::{AnnotationSource, ConstraintCategory, Location, ReturnConstraint};
use rustc_middle::ty::{self, RegionVid, TyCtxt};
use rustc_span::{DUMMY_SP, DesugaringKind};
use tracing::{debug, instrument, trace};

use crate::constraints::OutlivesConstraintSet;
use crate::constraints::graph::NormalConstraintGraph;
use crate::consumers::OutlivesConstraint;
use crate::handle_placeholders::RegionDefinitions;
use crate::region_infer::values::{LivenessValues, RegionElement};
use crate::type_check::Locations;

#[derive(Clone, Debug)]
pub(crate) struct BlameConstraint<'tcx> {
    pub category: ConstraintCategory<'tcx>,
    pub from_closure: bool,
    pub cause: ObligationCause<'tcx>,
    pub variance_info: ty::VarianceDiagInfo<TyCtxt<'tcx>>,
}

pub(crate) struct ConstraintSearch<'a, 'tcx> {
    pub(crate) definitions: &'a RegionDefinitions<'tcx>,
    pub(crate) fr_static: RegionVid,
    pub(crate) constraint_graph: &'a NormalConstraintGraph,
    pub(crate) constraints: &'a OutlivesConstraintSet<'tcx>,
}

impl<'a, 'tcx> ConstraintSearch<'a, 'tcx> {
    /// Get the region outlived by `longer_fr` and live at `element`.
    pub(crate) fn region_from_element(
        &self,
        liveness_constraints: &LivenessValues,
        longer_fr: RegionVid,
        element: &RegionElement<'tcx>,
    ) -> RegionVid {
        match *element {
            RegionElement::Location(l) => {
                self.find_sub_region_live_at(liveness_constraints, longer_fr, l)
            }
            RegionElement::RootUniversalRegion(r) => r,
            RegionElement::PlaceholderRegion(error_placeholder) => self
                .definitions
                .iter_enumerated()
                .find_map(|(r, definition)| match definition.origin {
                    NllRegionVariableOrigin::Placeholder(p) if p == error_placeholder => Some(r),
                    _ => None,
                })
                .unwrap(),
        }
    }

    /// Finds some region R such that `fr1: R` and `R` is live at `location`.
    #[instrument(skip(self, liveness_constraints), level = "trace", ret)]
    pub(crate) fn find_sub_region_live_at(
        &self,
        liveness_constraints: &LivenessValues,
        fr1: RegionVid,
        location: Location,
    ) -> RegionVid {
        self.constraint_path_to(
            fr1,
            |r| {
                trace!(?r, liveness_constraints=?liveness_constraints.pretty_print_live_points(r));
                liveness_constraints.is_live_at(r, location)
            },
            true,
        )
        .unwrap()
        .1
    }

    /// Walks the graph of constraints (where `'a: 'b` is considered
    /// an edge `'a -> 'b`) to find a path from `from_region` to
    /// `to_region`.
    ///
    /// Returns: a series of constraints as well as the region `R`
    /// that passed the target test.
    /// If `include_static_outlives_all` is `true`, then the synthetic
    /// outlives constraints `'static -> a` for every region `a` are
    /// considered in the search, otherwise they are ignored.
    #[instrument(skip(self, target_test), ret)]
    pub(crate) fn constraint_path_to(
        &self,
        from_region: RegionVid,
        target_test: impl Fn(RegionVid) -> bool,
        include_placeholder_static: bool,
    ) -> Option<(Vec<OutlivesConstraint<'tcx>>, RegionVid)> {
        self.find_constraint_path_between_regions_inner(
            true,
            from_region,
            &target_test,
            include_placeholder_static,
        )
        .or_else(|| {
            self.find_constraint_path_between_regions_inner(
                false,
                from_region,
                &target_test,
                include_placeholder_static,
            )
        })
    }

    pub(crate) fn constraint_path_between_regions(
        &self,
        from_region: RegionVid,
        to_region: RegionVid,
    ) -> Option<Vec<OutlivesConstraint<'tcx>>> {
        if from_region == to_region {
            bug!("Tried to find a path between {from_region:?} and itself!");
        }
        self.constraint_path_to(from_region, |t| t == to_region, true).map(|o| o.0)
    }

    /// The constraints we get from equating the hidden type of each use of an opaque
    /// with its final hidden type may end up getting preferred over other, potentially
    /// longer constraint paths.
    ///
    /// Given that we compute the final hidden type by relying on this existing constraint
    /// path, this can easily end up hiding the actual reason for why we require these regions
    /// to be equal.
    ///
    /// To handle this, we first look at the path while ignoring these constraints and then
    /// retry while considering them. This is not perfect, as the `from_region` may have already
    /// been partially related to its argument region, so while we rely on a member constraint
    /// to get a complete path, the most relevant step of that path already existed before then.
    fn find_constraint_path_between_regions_inner(
        &self,
        ignore_opaque_type_constraints: bool,
        from_region: RegionVid,
        target_test: impl Fn(RegionVid) -> bool,
        include_placeholder_static: bool,
    ) -> Option<(Vec<OutlivesConstraint<'tcx>>, RegionVid)> {
        let mut context = IndexVec::from_elem(Trace::NotVisited, self.definitions);
        context[from_region] = Trace::StartRegion;
        let fr_static = self.fr_static;

        // Use a deque so that we do a breadth-first search. We will
        // stop at the first match, which ought to be the shortest
        // path (fewest constraints).
        let mut deque = VecDeque::new();
        deque.push_back(from_region);

        while let Some(r) = deque.pop_front() {
            debug!("constraint_path_to: from_region={:?} r={:?}", from_region, r,);

            // Check if we reached the region we were looking for. If so,
            // we can reconstruct the path that led to it and return it.
            if target_test(r) {
                let mut result = vec![];
                let mut p = r;
                // This loop is cold and runs at the end, which is why we delay
                // `OutlivesConstraint` construction until now.
                loop {
                    match context[p] {
                        Trace::FromGraph(c) => {
                            p = c.sup;
                            result.push(*c);
                        }

                        Trace::FromStatic(sub) => {
                            let c = OutlivesConstraint {
                                sup: fr_static,
                                sub,
                                locations: Locations::All(DUMMY_SP),
                                span: DUMMY_SP,
                                category: ConstraintCategory::Internal,
                                variance_info: ty::VarianceDiagInfo::default(),
                                from_closure: false,
                            };
                            p = c.sup;
                            result.push(c);
                        }

                        Trace::StartRegion => {
                            result.reverse();
                            return Some((result, r));
                        }

                        Trace::NotVisited => {
                            bug!("found unvisited region {:?} on path to {:?}", p, r)
                        }
                    }
                }
            }

            // Otherwise, walk over the outgoing constraints and
            // enqueue any regions we find, keeping track of how we
            // reached them.

            // A constraint like `'r: 'x` can come from our constraint
            // graph.

            // Always inline this closure because it can be hot.
            let mut handle_trace = #[inline(always)]
            |sub, trace| {
                if let Trace::NotVisited = context[sub] {
                    context[sub] = trace;
                    deque.push_back(sub);
                }
            };

            // If this is the `'static` region and the graph's direction is normal, then set up the
            // Edges iterator to return all regions (#53178).
            if r == fr_static && self.constraint_graph.is_normal() {
                for sub in self.constraint_graph.outgoing_edges_from_static() {
                    handle_trace(sub, Trace::FromStatic(sub));
                }
            } else {
                let edges = self.constraint_graph.outgoing_edges_from_graph(r, self.constraints);
                // This loop can be hot.
                for constraint in edges {
                    match constraint.category {
                        ConstraintCategory::OutlivesUnnameablePlaceholder(_)
                            if !include_placeholder_static =>
                        {
                            debug!("Ignoring illegal placeholder constraint: {constraint:?}");
                            continue;
                        }
                        ConstraintCategory::OpaqueType if ignore_opaque_type_constraints => {
                            debug!("Ignoring member constraint: {constraint:?}");
                            continue;
                        }
                        _ => {}
                    }

                    debug_assert_eq!(constraint.sup, r);
                    handle_trace(constraint.sub, Trace::FromGraph(constraint));
                }
            }
        }

        None
    }

    /// Tries to find the best constraint to blame for the fact that
    /// `R: from_region`, where `R` is some region that meets
    /// `target_test`. This works by following the constraint graph,
    /// creating a constraint path that forces `R` to outlive
    /// `from_region`, and then finding the best choices within that
    /// path to blame.
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn best_blame_constraint(
        &self,
        from_region: RegionVid,
        from_region_origin: NllRegionVariableOrigin<'tcx>,
        to_region: RegionVid,
    ) -> (BlameConstraint<'tcx>, Vec<OutlivesConstraint<'tcx>>) {
        assert!(from_region != to_region, "Trying to blame a region for itself!");

        let path = self.constraint_path_to(from_region, |t| t == to_region, true).unwrap().0;

        // If we are passing through a constraint added because we reached an unnameable placeholder `'unnameable`,
        // redirect search towards `'unnameable`.
        let due_to_placeholder_outlives = path.iter().find_map(|c| {
            if let ConstraintCategory::OutlivesUnnameablePlaceholder(unnameable) = c.category {
                Some(unnameable)
            } else {
                None
            }
        });

        // Edge case: it's possible that `'from_region` is an unnameable placeholder.
        let path = if let Some(unnameable) = due_to_placeholder_outlives
            && unnameable != from_region
        {
            // We ignore the extra edges due to unnameable placeholders to get
            // an explanation that was present in the original constraint graph.
            self.constraint_path_to(from_region, |t| t == unnameable, false).unwrap().0
        } else {
            path
        };

        // We try to avoid reporting a `ConstraintCategory::Predicate` as our best constraint.
        // Instead, we use it to produce an improved `ObligationCauseCode`.
        // FIXME - determine what we should do if we encounter multiple
        // `ConstraintCategory::Predicate` constraints. Currently, we just pick the first one.
        let cause_code = path
            .iter()
            .find_map(|constraint| {
                if let ConstraintCategory::Predicate(predicate_span) = constraint.category {
                    // We currently do not store the `DefId` in the `ConstraintCategory`
                    // for performances reasons. The error reporting code used by NLL only
                    // uses the span, so this doesn't cause any problems at the moment.
                    Some(ObligationCauseCode::WhereClause(CRATE_DEF_ID.to_def_id(), predicate_span))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| ObligationCauseCode::Misc);

        // When reporting an error, there is typically a chain of constraints leading from some
        // "source" region which must outlive some "target" region.
        // In most cases, we prefer to "blame" the constraints closer to the target --
        // but there is one exception. When constraints arise from higher-ranked subtyping,
        // we generally prefer to blame the source value,
        // as the "target" in this case tends to be some type annotation that the user gave.
        // Therefore, if we find that the region origin is some instantiation
        // of a higher-ranked region, we start our search from the "source" point
        // rather than the "target", and we also tweak a few other things.
        //
        // An example might be this bit of Rust code:
        //
        // ```rust
        // let x: fn(&'static ()) = |_| {};
        // let y: for<'a> fn(&'a ()) = x;
        // ```
        //
        // In MIR, this will be converted into a combination of assignments and type ascriptions.
        // In particular, the 'static is imposed through a type ascription:
        //
        // ```rust
        // x = ...;
        // AscribeUserType(x, fn(&'static ())
        // y = x;
        // ```
        //
        // We wind up ultimately with constraints like
        //
        // ```rust
        // !a: 'temp1 // from the `y = x` statement
        // 'temp1: 'temp2
        // 'temp2: 'static // from the AscribeUserType
        // ```
        //
        // and here we prefer to blame the source (the y = x statement).
        let blame_source = match from_region_origin {
            NllRegionVariableOrigin::FreeRegion => true,
            NllRegionVariableOrigin::Placeholder(_) => false,
            // `'existential: 'whatever` never results in a region error by itself.
            // We may always infer it to `'static` afterall. This means while an error
            // path may go through an existential, these existentials are never the
            // `from_region`.
            NllRegionVariableOrigin::Existential { name: _ } => {
                unreachable!("existentials can outlive everything")
            }
        };

        // To pick a constraint to blame, we organize constraints by how interesting we expect them
        // to be in diagnostics, then pick the most interesting one closest to either the source or
        // the target on our constraint path.
        let constraint_interest = |constraint: &OutlivesConstraint<'tcx>| {
            use AnnotationSource::*;
            use ConstraintCategory::*;
            // Try to avoid blaming constraints from desugarings, since they may not clearly match
            // match what users have written. As an exception, allow blaming returns generated by
            // `?` desugaring, since the correspondence is fairly clear.
            let category = if let Some(kind) = constraint.span.desugaring_kind()
                && (kind != DesugaringKind::QuestionMark
                    || !matches!(constraint.category, Return(_)))
            {
                Boring
            } else {
                constraint.category
            };

            let interest = match category {
                // Returns usually provide a type to blame and have specially written diagnostics,
                // so prioritize them.
                Return(_) => 0,
                // Unsizing coercions are interesting, since we have a note for that:
                // `BorrowExplanation::add_object_lifetime_default_note`.
                // FIXME(dianne): That note shouldn't depend on a coercion being blamed; see issue
                // #131008 for an example of where we currently don't emit it but should.
                // Once the note is handled properly, this case should be removed. Until then, it
                // should be as limited as possible; the note is prone to false positives and this
                // constraint usually isn't best to blame.
                Cast {
                    is_raw_ptr_dyn_type_cast: _,
                    unsize_to: Some(unsize_ty),
                    is_implicit_coercion: true,
                } if to_region == self.fr_static
                    // Mirror the note's condition, to minimize how often this diverts blame.
                    && let ty::Adt(_, args) = unsize_ty.kind()
                    && args.iter().any(|arg| arg.as_type().is_some_and(|ty| ty.is_trait()))
                    // Mimic old logic for this, to minimize false positives in tests.
                    && !path
                        .iter()
                        .any(|c| matches!(c.category, TypeAnnotation(_))) =>
                {
                    1
                }
                // Between other interesting constraints, order by their position on the `path`.
                Yield
                | UseAsConst
                | UseAsStatic
                | TypeAnnotation(Ascription | Declaration | OpaqueCast)
                | Cast { .. }
                | CallArgument(_)
                | CopyBound
                | SizedBound
                | Assignment
                | Usage
                | ClosureUpvar(_) => 2,
                // Generic arguments are unlikely to be what relates regions together
                TypeAnnotation(GenericArg) => 3,
                // We handle predicates and opaque types specially; don't prioritize them here.
                Predicate(_) | OpaqueType => 4,
                // `Boring` constraints can correspond to user-written code and have useful spans,
                // but don't provide any other useful information for diagnostics.
                Boring => 5,
                // `BoringNoLocation` constraints can point to user-written code, but are less
                // specific, and are not used for relations that would make sense to blame.
                BoringNoLocation => 6,
                // Do not blame internal constraints if we can avoid it. Never blame
                // the `'region: 'static` constraints introduced by placeholder outlives.
                Internal => 7,
                OutlivesUnnameablePlaceholder(_) => 8,
            };

            debug!("constraint {constraint:?} category: {category:?}, interest: {interest:?}");

            interest
        };

        let best_choice = if blame_source {
            path.iter().enumerate().rev().min_by_key(|(_, c)| constraint_interest(c)).unwrap().0
        } else {
            path.iter().enumerate().min_by_key(|(_, c)| constraint_interest(c)).unwrap().0
        };

        debug!(?best_choice, ?blame_source);

        let best_constraint = if let Some(next) = path.get(best_choice + 1)
            && matches!(path[best_choice].category, ConstraintCategory::Return(_))
            && next.category == ConstraintCategory::OpaqueType
        {
            // The return expression is being influenced by the return type being
            // impl Trait, point at the return type and not the return expr.
            *next
        } else if path[best_choice].category == ConstraintCategory::Return(ReturnConstraint::Normal)
            && let Some(field) = path.iter().find_map(|p| {
                if let ConstraintCategory::ClosureUpvar(f) = p.category { Some(f) } else { None }
            })
        {
            OutlivesConstraint {
                category: ConstraintCategory::Return(ReturnConstraint::ClosureUpvar(field)),
                ..path[best_choice]
            }
        } else {
            path[best_choice]
        };

        assert!(
            !matches!(
                best_constraint.category,
                ConstraintCategory::OutlivesUnnameablePlaceholder(_)
            ),
            "Illegal placeholder constraint blamed; should have redirected to other region relation"
        );

        let blame_constraint = BlameConstraint {
            category: best_constraint.category,
            from_closure: best_constraint.from_closure,
            cause: ObligationCause::new(best_constraint.span, CRATE_DEF_ID, cause_code.clone()),
            variance_info: best_constraint.variance_info,
        };
        (blame_constraint, path)
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum Trace<'a, 'tcx> {
    StartRegion,
    FromGraph(&'a OutlivesConstraint<'tcx>),
    FromStatic(RegionVid),
    NotVisited,
}
