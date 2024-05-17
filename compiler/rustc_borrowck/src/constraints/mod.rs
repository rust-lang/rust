use crate::region_infer::RegionDefinition;
use crate::type_check::Locations;
use rustc_data_structures::graph::scc::{self, Sccs};
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::ty::{RegionVid, TyCtxt, UniverseIndex, VarianceDiagInfo};
use rustc_span::Span;
use std::fmt;
use std::ops::Index;

pub(crate) mod graph;

pub type ConstraintSccs = Sccs<RegionVid, ConstraintSccIndex, RegionTracker>;

/// An annotation for region graph SCCs that tracks
/// the values of its elements.
#[derive(Copy, Debug, Clone)]
pub struct RegionTracker {
    /// The largest universe of a placeholder reached from this SCC.
    /// This includes placeholders within this SCC.
    max_placeholder_universe_reached: UniverseIndex,

    /// The smallest universe index reachable form the nodes of this SCC.
    min_reachable_universe: UniverseIndex,

    /// The representative Region Variable Id for this SCC. We prefer
    /// placeholders over existentially quantified variables, otherwise
    ///  it's the one with the smallest Region Variable ID.
    pub representative: RegionVid,

    /// Is the current representative a placeholder?
    representative_is_placeholder: bool,

    /// Is the current representative existentially quantified?
    representative_is_existential: bool,
}

impl scc::Annotation for RegionTracker {
    fn merge_scc(mut self, mut other: Self) -> Self {
        // Prefer any placeholder over any existential
        if other.representative_is_placeholder && self.representative_is_existential {
            other.merge_min_max_seen(&self);
            return other;
        }

        if self.representative_is_placeholder && other.representative_is_existential
            || (self.representative <= other.representative)
        {
            self.merge_min_max_seen(&other);
            return self;
        }
        other.merge_min_max_seen(&self);
        other
    }

    fn merge_reached(mut self, other: Self) -> Self {
        // No update to in-component values, only add seen values.
        self.merge_min_max_seen(&other);
        self
    }
}

impl RegionTracker {
    pub fn new(rvid: RegionVid, definition: &RegionDefinition<'_>) -> Self {
        let (representative_is_placeholder, representative_is_existential) = match definition.origin {
            rustc_infer::infer::NllRegionVariableOrigin::FreeRegion => (false, false),
            rustc_infer::infer::NllRegionVariableOrigin::Placeholder(_) => (true, false),
            rustc_infer::infer::NllRegionVariableOrigin::Existential { .. } => (false, true),
        };

        let placeholder_universe =
            if representative_is_placeholder { definition.universe } else { UniverseIndex::ROOT };


        Self {
            max_placeholder_universe_reached: placeholder_universe,
            min_reachable_universe: definition.universe,
            representative: rvid,
            representative_is_placeholder,
            representative_is_existential,
        }
    }
    pub fn universe(self) -> UniverseIndex {
        self.min_reachable_universe
    }

    fn merge_min_max_seen(&mut self, other: &Self) {
        self.max_placeholder_universe_reached = std::cmp::max(
            self.max_placeholder_universe_reached,
            other.max_placeholder_universe_reached,
        );

        self.min_reachable_universe =
            std::cmp::min(self.min_reachable_universe, other.min_reachable_universe);
    }

    /// Returns `true` if during the annotated SCC reaches a placeholder
    /// with a universe larger than the smallest reachable one, `false` otherwise.
    pub fn has_incompatible_universes(&self) -> bool {
        self.universe().cannot_name(self.max_placeholder_universe_reached)
    }
}

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
    pub struct OutlivesConstraintIndex {}
}

rustc_index::newtype_index! {
    #[orderable]
    #[debug_format = "ConstraintSccIndex({})"]
    pub struct ConstraintSccIndex {}
}
