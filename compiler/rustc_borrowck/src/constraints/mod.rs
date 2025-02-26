use std::fmt;
use std::ops::Index;

use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::ty::{RegionVid, TyCtxt, VarianceDiagInfo};
use rustc_span::Span;
use tracing::{debug, instrument};

use crate::region_infer::{ConstraintSccs, RegionDefinition, RegionTracker};
use crate::type_check::Locations;
use crate::universal_regions::UniversalRegions;

pub(crate) mod graph;

/// A set of NLL region constraints. These include "outlives"
/// constraints of the form `R1: R2`. Each constraint is identified by
/// a unique `OutlivesConstraintIndex` and you can index into the set
/// (`constraint_set[i]`) to access the constraint details.
#[derive(Clone, Debug, Default)]
pub(crate) struct OutlivesConstraintSet<'tcx> {
    outlives: IndexVec<OutlivesConstraintIndex, OutlivesConstraint<'tcx>>,
}

impl<'tcx> OutlivesConstraintSet<'tcx> {
    pub(crate) fn push(&mut self, constraint: OutlivesConstraint<'tcx>) {
        debug!("OutlivesConstraintSet::push({:?})", constraint);
        if constraint.sup == constraint.sub {
            // 'a: 'a is pretty uninteresting
            return;
        }
        self.outlives.push(constraint);
    }

    /// Constructs a "normal" graph from the constraint set; the graph makes it
    /// easy to find the constraints affecting a particular region.
    ///
    /// N.B., this graph contains a "frozen" view of the current
    /// constraints. Any new constraints added to the `OutlivesConstraintSet`
    /// after the graph is built will not be present in the graph.
    pub(crate) fn graph(&self, num_region_vars: usize) -> graph::NormalConstraintGraph {
        graph::ConstraintGraph::new(graph::Normal, self, num_region_vars)
    }

    /// Like `graph`, but constraints a reverse graph where `R1: R2`
    /// represents an edge `R2 -> R1`.
    pub(crate) fn reverse_graph(&self, num_region_vars: usize) -> graph::ReverseConstraintGraph {
        graph::ConstraintGraph::new(graph::Reverse, self, num_region_vars)
    }

    pub(crate) fn outlives(
        &self,
    ) -> &IndexSlice<OutlivesConstraintIndex, OutlivesConstraint<'tcx>> {
        &self.outlives
    }

    /// Computes cycles (SCCs) in the graph of regions. In particular,
    /// find all regions R1, R2 such that R1: R2 and R2: R1 and group
    /// them into an SCC, and find the relationships between SCCs.
    pub(crate) fn compute_sccs(
        &self,
        static_region: RegionVid,
        definitions: &IndexVec<RegionVid, RegionDefinition<'tcx>>,
    ) -> ConstraintSccs {
        let constraint_graph = self.graph(definitions.len());
        let region_graph = &constraint_graph.region_graph(self, static_region);
        ConstraintSccs::new_with_annotation(&region_graph, |r| {
            RegionTracker::new(r, &definitions[r])
        })
    }

    /// This method handles Universe errors by rewriting the constraint
    /// graph. For each strongly connected component in the constraint
    /// graph such that there is a series of constraints
    ///    A: B: C: ... : X  where
    /// A's universe is smaller than X's and A is a placeholder,
    /// add a constraint that A: 'static. This is a safe upper bound
    /// in the face of borrow checker/trait solver limitations that will
    /// eventually go away.
    ///
    /// For a more precise definition, see the documentation for
    /// [`RegionTracker::has_incompatible_universes()`].
    ///
    /// This edge case used to be handled during constraint propagation
    /// by iterating over the strongly connected components in the constraint
    /// graph while maintaining a set of bookkeeping mappings similar
    /// to what is stored in `RegionTracker` and manually adding 'sttaic as
    /// needed.
    ///
    /// It was rewritten as part of the Polonius project with the goal of moving
    /// higher-kindedness concerns out of the path of the borrow checker,
    /// for two reasons:
    ///
    /// 1. Implementing Polonius is difficult enough without also
    ///     handling them.
    /// 2. The long-term goal is to handle higher-kinded concerns
    ///     in the trait solver, where they belong. This avoids
    ///     logic duplication and allows future trait solvers
    ///     to compute better bounds than for example our
    ///     "must outlive 'static" here.
    ///
    /// This code is a stop-gap measure in preparation for the future trait solver.
    ///
    /// Every constraint added by this method is an
    /// internal `IllegalUniverse` constraint.
    #[instrument(skip(self, universal_regions, definitions))]
    pub(crate) fn add_outlives_static(
        &mut self,
        universal_regions: &UniversalRegions<'tcx>,
        definitions: &IndexVec<RegionVid, RegionDefinition<'tcx>>,
    ) -> ConstraintSccs {
        let fr_static = universal_regions.fr_static;
        let sccs = self.compute_sccs(fr_static, definitions);

        // Changed to `true` if we added any constraints to `self` and need to
        // recompute SCCs.
        let mut added_constraints = false;

        for scc in sccs.all_sccs() {
            // No point in adding 'static: 'static!
            // This micro-optimisation makes somewhat sense
            // because static outlives *everything*.
            if scc == sccs.scc(fr_static) {
                continue;
            }

            let annotation = sccs.annotation(scc);

            // If this SCC participates in a universe violation,
            // e.g. if it reaches a region with a universe smaller than
            // the largest region reached, add a requirement that it must
            // outlive `'static`.
            if annotation.has_incompatible_universes() {
                // Optimisation opportunity: this will add more constraints than
                // needed for correctness, since an SCC upstream of another with
                // a universe violation will "infect" its downstream SCCs to also
                // outlive static.
                added_constraints = true;
                let scc_representative_outlives_static = OutlivesConstraint {
                    sup: annotation.representative,
                    sub: fr_static,
                    category: ConstraintCategory::IllegalUniverse,
                    locations: Locations::All(rustc_span::DUMMY_SP),
                    span: rustc_span::DUMMY_SP,
                    variance_info: VarianceDiagInfo::None,
                    from_closure: false,
                };
                self.push(scc_representative_outlives_static);
            }
        }

        if added_constraints {
            // We changed the constraint set and so must recompute SCCs.
            self.compute_sccs(fr_static, definitions)
        } else {
            // If we didn't add any back-edges; no more work needs doing
            sccs
        }
    }
}

impl<'tcx> Index<OutlivesConstraintIndex> for OutlivesConstraintSet<'tcx> {
    type Output = OutlivesConstraint<'tcx>;

    fn index(&self, i: OutlivesConstraintIndex) -> &Self::Output {
        &self.outlives[i]
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct OutlivesConstraint<'tcx> {
    // NB. The ordering here is not significant for correctness, but
    // it is for convenience. Before we dump the constraints in the
    // debugging logs, we sort them, and we'd like the "super region"
    // to be first, etc. (In particular, span should remain last.)
    /// The region SUP must outlive SUB...
    pub sup: RegionVid,

    /// Region that must be outlived.
    pub sub: RegionVid,

    /// Where did this constraint arise?
    pub locations: Locations,

    /// The `Span` associated with the creation of this constraint.
    /// This should be used in preference to obtaining the span from
    /// `locations`, since the `locations` may give a poor span
    /// in some cases (e.g. converting a constraint from a promoted).
    pub span: Span,

    /// What caused this constraint?
    pub category: ConstraintCategory<'tcx>,

    /// Variance diagnostic information
    pub variance_info: VarianceDiagInfo<TyCtxt<'tcx>>,

    /// If this constraint is promoted from closure requirements.
    pub from_closure: bool,
}

impl<'tcx> fmt::Debug for OutlivesConstraint<'tcx> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "({:?}: {:?}) due to {:?} ({:?}) ({:?})",
            self.sup, self.sub, self.locations, self.variance_info, self.category,
        )
    }
}

rustc_index::newtype_index! {
    #[debug_format = "OutlivesConstraintIndex({})"]
    pub(crate) struct OutlivesConstraintIndex {}
}

rustc_index::newtype_index! {
    #[orderable]
    #[debug_format = "ConstraintSccIndex({})"]
    pub struct ConstraintSccIndex {}
}
