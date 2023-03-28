use super::*;
use crate::infer::CombinedSnapshot;
use rustc_data_structures::{
    fx::FxIndexMap,
    graph::{scc::Sccs, vec_graph::VecGraph},
    undo_log::UndoLogs,
};
use rustc_index::vec::Idx;
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::relate::RelateResult;

impl<'tcx> RegionConstraintCollector<'_, 'tcx> {
    /// Searches new universes created during `snapshot`, looking for
    /// placeholders that may "leak" out from the universes they are contained
    /// in. If any leaking placeholders are found, then an `Err` is returned
    /// (typically leading to the snapshot being reversed).
    ///
    /// The leak check *used* to be the only way we had to handle higher-ranked
    /// obligations. Now that we have integrated universes into the region
    /// solvers, this is no longer the case, but we retain the leak check for
    /// backwards compatibility purposes. In particular, it lets us make "early"
    /// decisions about whether a region error will be reported that are used in
    /// coherence and elsewhere -- see #56105 and #59490 for more details. The
    /// eventual fate of the leak checker is not yet settled.
    ///
    /// The leak checker works by searching for the following error patterns:
    ///
    /// * P1: P2, where P1 != P2
    /// * P1: R, where R is in some universe that cannot name P1
    ///
    /// The idea here is that each of these patterns represents something that
    /// the region solver would eventually report as an error, so we can detect
    /// the error early. There is a fly in the ointment, though, in that this is
    /// not entirely true. In particular, in the future, we may extend the
    /// environment with implied bounds or other info about how placeholders
    /// relate to regions in outer universes. In that case, `P1: R` for example
    /// might become solvable.
    ///
    /// # Summary of the implementation
    ///
    /// The leak checks as follows. First, we construct a graph where `R2: R1`
    /// implies `R2 -> R1`, and we compute the SCCs.
    ///
    /// For each SCC S, we compute:
    ///
    /// * what placeholder P it must be equal to, if any
    ///   * if there are multiple placeholders that must be equal, report an error because `P1: P2`
    /// * the minimum universe of its constituents
    ///
    /// Then we walk the SCCs in dependency order and compute
    ///
    /// * what placeholder they must outlive transitively
    ///   * if they must also be equal to a placeholder, report an error because `P1: P2`
    /// * minimum universe U of all SCCs they must outlive
    ///   * if they must also be equal to a placeholder P, and U cannot name P, report an error, as that
    ///     indicates `P: R` and `R` is in an incompatible universe
    ///
    /// # Historical note
    ///
    /// Older variants of the leak check used to report errors for these
    /// patterns, but we no longer do:
    ///
    /// * R: P1, even if R cannot name P1, because R = 'static is a valid sol'n
    /// * R: P1, R: P2, as above
    pub fn leak_check(
        &mut self,
        tcx: TyCtxt<'tcx>,
        overly_polymorphic: bool,
        max_universe: ty::UniverseIndex,
        snapshot: &CombinedSnapshot<'tcx>,
    ) -> RelateResult<'tcx, ()> {
        debug!(
            "leak_check(max_universe={:?}, snapshot.universe={:?}, overly_polymorphic={:?})",
            max_universe, snapshot.universe, overly_polymorphic
        );

        assert!(UndoLogs::<super::UndoLog<'_>>::in_snapshot(&self.undo_log));

        let universe_at_start_of_snapshot = snapshot.universe;
        if universe_at_start_of_snapshot == max_universe {
            return Ok(());
        }

        let mini_graph =
            &MiniGraph::new(tcx, self.undo_log.region_constraints(), &self.storage.data.verifys);

        let mut leak_check = LeakCheck::new(
            tcx,
            universe_at_start_of_snapshot,
            max_universe,
            overly_polymorphic,
            mini_graph,
            self,
        );
        leak_check.assign_placeholder_values()?;
        leak_check.propagate_scc_value()?;
        Ok(())
    }
}

struct LeakCheck<'me, 'tcx> {
    tcx: TyCtxt<'tcx>,
    universe_at_start_of_snapshot: ty::UniverseIndex,
    /// Only used when reporting region errors.
    overly_polymorphic: bool,
    mini_graph: &'me MiniGraph<'tcx>,
    rcc: &'me RegionConstraintCollector<'me, 'tcx>,

    // Initially, for each SCC S, stores a placeholder `P` such that `S = P`
    // must hold.
    //
    // Later, during the [`LeakCheck::propagate_scc_value`] function, this array
    // is repurposed to store some placeholder `P` such that the weaker
    // condition `S: P` must hold. (This is true if `S: S1` transitively and `S1
    // = P`.)
    scc_placeholders: IndexVec<LeakCheckScc, Option<ty::PlaceholderRegion>>,

    // For each SCC S, track the minimum universe that flows into it. Note that
    // this is both the minimum of the universes for every region that is a
    // member of the SCC, but also if you have `R1: R2`, then the universe of
    // `R2` must be less than the universe of `R1` (i.e., `R1` flows `R2`). To
    // see that, imagine that you have `P1: R` -- in that case, `R` must be
    // either the placeholder `P1` or the empty region in that same universe.
    //
    // To detect errors, we look for an SCC S where the values in
    // `scc_values[S]` (if any) cannot be stored into `scc_universes[S]`.
    scc_universes: IndexVec<LeakCheckScc, SccUniverse<'tcx>>,
}

impl<'me, 'tcx> LeakCheck<'me, 'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        universe_at_start_of_snapshot: ty::UniverseIndex,
        max_universe: ty::UniverseIndex,
        overly_polymorphic: bool,
        mini_graph: &'me MiniGraph<'tcx>,
        rcc: &'me RegionConstraintCollector<'me, 'tcx>,
    ) -> Self {
        let dummy_scc_universe = SccUniverse { universe: max_universe, region: None };
        Self {
            tcx,
            universe_at_start_of_snapshot,
            overly_polymorphic,
            mini_graph,
            rcc,
            scc_placeholders: IndexVec::from_elem_n(None, mini_graph.sccs.num_sccs()),
            scc_universes: IndexVec::from_elem_n(dummy_scc_universe, mini_graph.sccs.num_sccs()),
        }
    }

    /// Compute what placeholders (if any) each SCC must be equal to.
    /// Also compute the minimum universe of all the regions in each SCC.
    fn assign_placeholder_values(&mut self) -> RelateResult<'tcx, ()> {
        // First walk: find each placeholder that is from a newly created universe.
        for (region, leak_check_node) in &self.mini_graph.nodes {
            let scc = self.mini_graph.sccs.scc(*leak_check_node);

            // Set the universe of each SCC to be the minimum of its constituent universes
            let universe = self.rcc.universe(*region);
            debug!(
                "assign_placeholder_values: scc={:?} universe={:?} region={:?}",
                scc, universe, region
            );
            self.scc_universes[scc].take_min(universe, *region);

            // Detect those SCCs that directly contain a placeholder
            if let ty::RePlaceholder(placeholder) = **region {
                if self.universe_at_start_of_snapshot.cannot_name(placeholder.universe) {
                    self.assign_scc_value(scc, placeholder)?;
                }
            }
        }

        Ok(())
    }

    // assign_scc_value(S, P): Update `scc_values` to account for the fact that `P: S` must hold.
    // This may create an error.
    fn assign_scc_value(
        &mut self,
        scc: LeakCheckScc,
        placeholder: ty::PlaceholderRegion,
    ) -> RelateResult<'tcx, ()> {
        match self.scc_placeholders[scc] {
            Some(p) => {
                assert_ne!(p, placeholder);
                return Err(self.placeholder_error(p, placeholder));
            }
            None => {
                self.scc_placeholders[scc] = Some(placeholder);
            }
        };

        Ok(())
    }

    /// For each SCC S, iterate over each successor S1 where `S: S1`:
    ///
    /// * Compute
    /// Iterate over each SCC `S` and ensure that, for each `S1` where `S1: S`,
    /// `universe(S) <= universe(S1)`. This executes after
    /// `assign_placeholder_values`, so `universe(S)` is already the minimum
    /// universe of any of its direct constituents.
    fn propagate_scc_value(&mut self) -> RelateResult<'tcx, ()> {
        // Loop invariants:
        //
        // On start of the loop iteration for `scc1`:
        //
        // * `scc_universes[scc1]` contains the minimum universe of the
        //   constituents of `scc1`
        // * `scc_placeholder[scc1]` stores the placeholder that `scc1` must
        //   be equal to (if any)
        //
        // For each successor `scc2` where `scc1: scc2`:
        //
        // * `scc_placeholder[scc2]` stores some placeholder `P` where
        //   `scc2: P` (if any)
        // * `scc_universes[scc2]` contains the minimum universe of the
        //   constituents of `scc2` and any of its successors
        for scc1 in self.mini_graph.sccs.all_sccs() {
            debug!(
                "propagate_scc_value: scc={:?} with universe {:?}",
                scc1, self.scc_universes[scc1]
            );

            // Walk over each `scc2` such that `scc1: scc2` and compute:
            //
            // * `scc1_universe`: the minimum universe of `scc2` and the constituents of `scc1`
            // * `succ_bound`: placeholder `P` that the successors must outlive, if any (if there are multiple,
            //   we pick one arbitrarily)
            let mut scc1_universe = self.scc_universes[scc1];
            let mut succ_bound = None;
            for &scc2 in self.mini_graph.sccs.successors(scc1) {
                let SccUniverse { universe: scc2_universe, region: scc2_region } =
                    self.scc_universes[scc2];

                scc1_universe.take_min(scc2_universe, scc2_region.unwrap());

                if let Some(b) = self.scc_placeholders[scc2] {
                    succ_bound = Some(b);
                }
            }

            // Update minimum universe of scc1.
            self.scc_universes[scc1] = scc1_universe;

            // At this point, `scc_placeholders[scc1]` stores the placeholder that
            // `scc1` must be equal to, if any.
            if let Some(scc1_placeholder) = self.scc_placeholders[scc1] {
                debug!(
                    "propagate_scc_value: scc1={:?} placeholder={:?} scc1_universe={:?}",
                    scc1, scc1_placeholder, scc1_universe
                );

                // Check if `P1: R` for some `R` in a universe that cannot name
                // P1. That's an error.
                if scc1_universe.universe.cannot_name(scc1_placeholder.universe) {
                    return Err(self.error(scc1_placeholder, scc1_universe.region.unwrap()));
                }

                // Check if we have some placeholder where `S: P2`
                // (transitively). In that case, since `S = P1`, that implies
                // `P1: P2`, which is an error condition.
                if let Some(scc2_placeholder) = succ_bound {
                    assert_ne!(scc1_placeholder, scc2_placeholder);
                    return Err(self.placeholder_error(scc1_placeholder, scc2_placeholder));
                }
            } else {
                // Otherwise, we can reach a placeholder if some successor can.
                self.scc_placeholders[scc1] = succ_bound;
            }

            // At this point, `scc_placeholder[scc1]` stores some placeholder that `scc1` must outlive (if any).
        }
        Ok(())
    }

    fn placeholder_error(
        &self,
        placeholder1: ty::PlaceholderRegion,
        placeholder2: ty::PlaceholderRegion,
    ) -> TypeError<'tcx> {
        self.error(placeholder1, self.tcx.mk_re_placeholder(placeholder2))
    }

    fn error(
        &self,
        placeholder: ty::PlaceholderRegion,
        other_region: ty::Region<'tcx>,
    ) -> TypeError<'tcx> {
        debug!("error: placeholder={:?}, other_region={:?}", placeholder, other_region);
        if self.overly_polymorphic {
            TypeError::RegionsOverlyPolymorphic(placeholder.name, other_region)
        } else {
            TypeError::RegionsInsufficientlyPolymorphic(placeholder.name, other_region)
        }
    }
}

// States we need to distinguish:
//
// * must be equal to a placeholder (i.e., a placeholder is in the SCC)
//     * it could conflict with some other regions in the SCC in different universes
//     * or a different placeholder
// * `P1: S` and `S` must be equal to a placeholder
// * `P1: S` and `S` is in an incompatible universe
//
// So if we
//
// (a) compute which placeholder (if any) each SCC must be equal to
// (b) compute its minimum universe
// (c) compute *some* placeholder where `S: P1` (any one will do)
//
// then we get an error if:
//
// - it must be equal to a placeholder `P1` and minimum universe cannot name `P1`
// - `S: P1` and minimum universe cannot name `P1`
// - `S: P1` and we must be equal to `P2`
//
// So we want to track:
//
// * Equal placeholder (if any)
// * Some bounding placeholder (if any)
// * Minimum universe
//
// * We compute equal placeholder + minimum universe of constituents in first pass
// * Then we walk in order and compute from our dependencies `S1` where `S: S1` (`S -> S1`)
//   * bounding placeholder (if any)
//   * minimum universe
// * And if we must be equal to a placeholder then we check it against
//   * minimum universe
//   * no bounding placeholder

/// Tracks the "minimum universe" for each SCC, along with some region that
/// caused it to change.
#[derive(Copy, Clone, Debug)]
struct SccUniverse<'tcx> {
    /// For some SCC S, the minimum universe of:
    ///
    /// * each region R in S
    /// * each SCC S1 such that S: S1
    universe: ty::UniverseIndex,

    /// Some region that caused `universe` to be what it is.
    region: Option<ty::Region<'tcx>>,
}

impl<'tcx> SccUniverse<'tcx> {
    /// If `universe` is less than our current universe, then update
    /// `self.universe` and `self.region`.
    fn take_min(&mut self, universe: ty::UniverseIndex, region: ty::Region<'tcx>) {
        if universe < self.universe || self.region.is_none() {
            self.universe = universe;
            self.region = Some(region);
        }
    }
}

rustc_index::newtype_index! {
    #[debug_format = "LeakCheckNode({})"]
    struct LeakCheckNode {}
}

rustc_index::newtype_index! {
    #[debug_format = "LeakCheckScc({})"]
    struct LeakCheckScc {}
}

/// Represents the graph of constraints. For each `R1: R2` constraint we create
/// an edge `R1 -> R2` in the graph.
struct MiniGraph<'tcx> {
    /// Map from a region to the index of the node in the graph.
    nodes: FxIndexMap<ty::Region<'tcx>, LeakCheckNode>,

    /// Map from node index to SCC, and stores the successors of each SCC. All
    /// the regions in the same SCC are equal to one another, and if `S1 -> S2`,
    /// then `S1: S2`.
    sccs: Sccs<LeakCheckNode, LeakCheckScc>,
}

impl<'tcx> MiniGraph<'tcx> {
    fn new<'a>(
        tcx: TyCtxt<'tcx>,
        undo_log: impl Iterator<Item = &'a UndoLog<'tcx>>,
        verifys: &[Verify<'tcx>],
    ) -> Self
    where
        'tcx: 'a,
    {
        let mut nodes = FxIndexMap::default();
        let mut edges = Vec::new();

        // Note that if `R2: R1`, we get a callback `r1, r2`, so `target` is first parameter.
        Self::iterate_undo_log(tcx, undo_log, verifys, |target, source| {
            let source_node = Self::add_node(&mut nodes, source);
            let target_node = Self::add_node(&mut nodes, target);
            edges.push((source_node, target_node));
        });
        let graph = VecGraph::new(nodes.len(), edges);
        let sccs = Sccs::new(&graph);
        Self { nodes, sccs }
    }

    /// Invokes `each_edge(R1, R2)` for each edge where `R2: R1`
    fn iterate_undo_log<'a>(
        tcx: TyCtxt<'tcx>,
        undo_log: impl Iterator<Item = &'a UndoLog<'tcx>>,
        verifys: &[Verify<'tcx>],
        mut each_edge: impl FnMut(ty::Region<'tcx>, ty::Region<'tcx>),
    ) where
        'tcx: 'a,
    {
        for undo_entry in undo_log {
            match undo_entry {
                &AddConstraint(Constraint::VarSubVar(a, b)) => {
                    each_edge(tcx.mk_re_var(a), tcx.mk_re_var(b));
                }
                &AddConstraint(Constraint::RegSubVar(a, b)) => {
                    each_edge(a, tcx.mk_re_var(b));
                }
                &AddConstraint(Constraint::VarSubReg(a, b)) => {
                    each_edge(tcx.mk_re_var(a), b);
                }
                &AddConstraint(Constraint::RegSubReg(a, b)) => {
                    each_edge(a, b);
                }
                &AddVerify(i) => span_bug!(
                    verifys[i].origin.span(),
                    "we never add verifications while doing higher-ranked things",
                ),
                &AddCombination(..) | &AddVar(..) => {}
            }
        }
    }

    fn add_node(
        nodes: &mut FxIndexMap<ty::Region<'tcx>, LeakCheckNode>,
        r: ty::Region<'tcx>,
    ) -> LeakCheckNode {
        let l = nodes.len();
        *nodes.entry(r).or_insert(LeakCheckNode::new(l))
    }
}
