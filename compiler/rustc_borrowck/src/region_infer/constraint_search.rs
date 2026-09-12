//! Searching the region graph.
use std::collections::VecDeque;

use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_middle::bug;
use rustc_middle::mir::{AnnotationSource, ConstraintCategory, Location, ReturnConstraint};
use rustc_middle::ty::{self, RegionVid};
use rustc_span::DUMMY_SP;
use rustc_span::hygiene::DesugaringKind;
use tracing::{debug, instrument};

use crate::constraints::OutlivesConstraintSet;
use crate::constraints::graph::NormalConstraintGraph;
use crate::consumers::OutlivesConstraint;
use crate::region_infer::values::LivenessValues;
use crate::region_infer::{BestBlame, ConstraintSccs};
use crate::type_check::Locations;

type Path<'tcx> = Vec<OutlivesConstraint<'tcx>>;

/// Represents an ongoing search through the outlives graph.
pub(crate) struct ConstraintSearch<'a, 'tcx> {
    outlives: &'a OutlivesConstraintSet<'tcx>,
    constraint_graph: &'a NormalConstraintGraph,
    constraint_sccs: &'a ConstraintSccs,
    fr_static: RegionVid,
    nr_regions: usize,
    from_region: RegionVid,
}

impl<'a, 'tcx> ConstraintSearch<'a, 'tcx> {
    /// Tries to find the best constraint to blame for the fact that
    /// `R: from_region`, where `R` is some region that meets
    /// `target_test`. This works by following the constraint graph,
    /// creating a constraint path that forces `R` to outlive
    /// `from_region`, and then finding the best choices within that
    /// path to blame.
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn best_blame_constraint(
        self,
        from_region_origin: NllRegionVariableOrigin<'tcx>,
        to_region: RegionVid,
    ) -> BestBlame<'tcx> {
        let from_region = self.from_region;
        assert!(self.from_region != to_region, "Trying to blame a region for itself!");
        let fr_static = self.fr_static;

        let path = self.to_region(to_region).unwrap();

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
        let mut path = if let Some(unnameable) = due_to_placeholder_outlives
            && unnameable != from_region
        {
            self.to_region(unnameable).unwrap()
        } else {
            path
        };

        debug!(
            "path={:#?}",
            path.iter()
                .map(|c| format!(
                    "{:?} ({:?}: {:?})",
                    c,
                    self.constraint_sccs.scc(c.sup),
                    self.constraint_sccs.scc(c.sub),
                ))
                .collect::<Vec<_>>()
        );

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
            // Try to avoid blaming constraints from desugarings, since they may not clearly match
            // match what users have written. As an exception, allow blaming returns generated by
            // `?` desugaring, since the correspondence is fairly clear.
            let category = if let Some(kind) = constraint.span.desugaring_kind()
                && (kind != DesugaringKind::QuestionMark
                    || !matches!(constraint.category, ConstraintCategory::Return(_)))
            {
                ConstraintCategory::Boring
            } else {
                constraint.category
            };

            let interest = match category {
                // Returns usually provide a type to blame and have specially written diagnostics,
                // so prioritize them.
                ConstraintCategory::Return(_) => 0,
                // Unsizing coercions are interesting, since we have a note for that:
                // `BorrowExplanation::add_object_lifetime_default_note`.
                // FIXME(dianne): That note shouldn't depend on a coercion being blamed; see issue
                // #131008 for an example of where we currently don't emit it but should.
                // Once the note is handled properly, this case should be removed. Until then, it
                // should be as limited as possible; the note is prone to false positives and this
                // constraint usually isn't best to blame.
                ConstraintCategory::Cast {
                    is_raw_ptr_dyn_type_cast: _,
                    unsize_to: Some(unsize_ty),
                    is_implicit_coercion: true,
                } if to_region == fr_static
                    // Mirror the note's condition, to minimize how often this diverts blame.
                    && let ty::Adt(_, args) = unsize_ty.kind()
                    && args.iter().any(|arg| arg.as_type().is_some_and(|ty| ty.is_trait()))
                    // Mimic old logic for this, to minimize false positives in tests.
                    && !path
                        .iter()
                        .any(|c| matches!(c.category, ConstraintCategory::TypeAnnotation(_))) =>
                {
                    1
                }
                // Between other interesting constraints, order by their position on the `path`.
                ConstraintCategory::Yield
                | ConstraintCategory::UseAsConst
                | ConstraintCategory::UseAsStatic
                | ConstraintCategory::TypeAnnotation(
                    AnnotationSource::Ascription
                    | AnnotationSource::Declaration
                    | AnnotationSource::OpaqueCast,
                )
                | ConstraintCategory::Cast { .. }
                | ConstraintCategory::CallArgument(_)
                | ConstraintCategory::CopyBound
                | ConstraintCategory::SizedBound
                | ConstraintCategory::Assignment
                | ConstraintCategory::Usage
                | ConstraintCategory::ClosureUpvar(_) => 2,
                // Generic arguments are unlikely to be what relates regions together
                ConstraintCategory::TypeAnnotation(AnnotationSource::GenericArg) => 3,
                // We handle predicates and opaque types specially; don't prioritize them here.
                ConstraintCategory::Predicate(_) | ConstraintCategory::OpaqueType => 4,
                // `Boring` constraints can correspond to user-written code and have useful spans,
                // but don't provide any other useful information for diagnostics.
                ConstraintCategory::Boring => 5,
                // `BoringNoLocation` constraints can point to user-written code, but are less
                // specific, and are not used for relations that would make sense to blame.
                ConstraintCategory::BoringNoLocation => 6,
                // Do not blame internal constraints if we can avoid it. Never blame
                // the `'region: 'static` constraints introduced by placeholder outlives.
                ConstraintCategory::Internal => 7,
                ConstraintCategory::OutlivesUnnameablePlaceholder(_) => 8,
                ConstraintCategory::SolverRegionConstraint(_) => 9,
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

        let best_blame_idx = if let Some(next) = path.get(best_choice + 1)
            && matches!(path[best_choice].category, ConstraintCategory::Return(_))
            && next.category == ConstraintCategory::OpaqueType
        {
            // The return expression is being influenced by the return type being
            // impl Trait, point at the return type and not the return expr.
            best_choice + 1
        } else if path[best_choice].category == ConstraintCategory::Return(ReturnConstraint::Normal)
            && let Some(field) = path.iter().find_map(|p| {
                if let ConstraintCategory::ClosureUpvar(f) = p.category { Some(f) } else { None }
            })
        {
            path[best_choice].category =
                ConstraintCategory::Return(ReturnConstraint::ClosureUpvar(field));
            best_choice
        } else {
            best_choice
        };

        assert!(
            !matches!(
                path[best_blame_idx].category,
                ConstraintCategory::OutlivesUnnameablePlaceholder(_)
            ),
            "Illegal placeholder constraint blamed; should have redirected to other region relation"
        );

        BestBlame { path, idx: best_blame_idx }
    }

    /// Walks the graph of constraints (where `'a: 'b` is considered an edge `'a
    /// -> 'b`) to find a path from `from_region` to either a specified region
    /// (`to_region()`) or a region live at some location (`live_at()`)
    pub(crate) fn begin(
        outlives: &'a OutlivesConstraintSet<'tcx>,
        constraint_graph: &'a NormalConstraintGraph,
        constraint_sccs: &'a ConstraintSccs,
        fr_static: RegionVid,
        nr_regions: usize,
        from_region: RegionVid,
    ) -> ConstraintSearch<'a, 'tcx> {
        Self { outlives, constraint_graph, constraint_sccs, fr_static, nr_regions, from_region }
    }

    /// Returns: a series of constraints visited on the way. If
    /// `include_static_outlives_all` is `true`, then the synthetic outlives
    /// constraints `'static -> a` for every region `a` are considered in the
    /// search, otherwise they are ignored.
    #[instrument(skip(self), ret)]
    pub(crate) fn to_region(&self, to_region: RegionVid) -> Option<Path<'tcx>> {
        self.find_constraint_path_between_regions_inner(true, self.from_region, to_region, false)
            .or_else(|| {
                self.find_constraint_path_between_regions_inner(
                    true,
                    self.from_region,
                    to_region,
                    true,
                )
            })
            .or_else(|| {
                self.find_constraint_path_between_regions_inner(
                    false,
                    self.from_region,
                    to_region,
                    true,
                )
            })
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
        target_region: RegionVid,
        include_placeholder_static: bool,
    ) -> Option<Path<'tcx>> {
        let mut context = IndexVec::from_elem_n(Trace::NotVisited, self.nr_regions);
        context[from_region] = Trace::StartRegion;

        let fr_static = self.fr_static;

        // Use a deque so that we do a breadth-first search. We will
        // stop at the first match, which ought to be the shortest
        // path (fewest constraints).
        let mut deque = VecDeque::new();
        deque.push_back(from_region);

        while let Some(r) = deque.pop_front() {
            // Check if we reached the region we were looking for. If so,
            // we can reconstruct the path that led to it and return it.
            if target_region == r {
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
                            return Some(result);
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
                let edges = self.constraint_graph.outgoing_edges_from_graph(r, &self.outlives);
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

    /// Search for an outlived region which is live at `location` and return it.
    pub(crate) fn live_at(
        self,
        liveness_constraints: &LivenessValues,
        location: Location,
    ) -> Option<RegionVid> {
        let mut visited = DenseBitSet::new_empty(self.nr_regions);
        visited.insert(self.from_region);

        let mut deque = VecDeque::new();
        deque.push_back(self.from_region);

        while let Some(r) = deque.pop_front() {
            if liveness_constraints.is_live_at(r, location) {
                return Some(r);
            }

            debug_assert!(
                r != self.fr_static,
                "'static should have been live; no implementation for walking from 'static!"
            );

            let edges = self.constraint_graph.outgoing_edges_from_graph(r, &self.outlives);
            // This loop can be hot.
            for constraint in edges {
                debug_assert_eq!(constraint.sup, r);
                if visited.insert(constraint.sub) {
                    deque.push_back(constraint.sub)
                }
            }
        }
        None
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum Trace<'a, 'tcx> {
    StartRegion,
    FromGraph(&'a OutlivesConstraint<'tcx>),
    FromStatic(RegionVid),
    NotVisited,
}
