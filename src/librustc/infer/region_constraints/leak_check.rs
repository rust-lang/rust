use super::*;
use crate::infer::{CombinedSnapshot, PlaceholderMap};
use crate::ty::error::TypeError;
use crate::ty::relate::RelateResult;

impl<'tcx> RegionConstraintCollector<'tcx> {
    /// Searches region constraints created since `snapshot` that
    /// affect one of the placeholders in `placeholder_map`, returning
    /// an error if any of the placeholders are related to another
    /// placeholder or would have to escape into some parent universe
    /// that cannot name them.
    ///
    /// This is a temporary backwards compatibility measure to try and
    /// retain the older (arguably incorrect) behavior of the
    /// compiler.
    ///
    /// NB. The use of snapshot here is mostly an efficiency thing --
    /// we could search *all* region constraints, but that'd be a
    /// bigger set and the data structures are not setup for that. If
    /// we wind up keeping some form of this check long term, it would
    /// probably be better to remove the snapshot parameter and to
    /// refactor the constraint set.
    pub fn leak_check(
        &mut self,
        tcx: TyCtxt<'tcx>,
        overly_polymorphic: bool,
        placeholder_map: &PlaceholderMap<'tcx>,
        _snapshot: &CombinedSnapshot<'_, 'tcx>,
    ) -> RelateResult<'tcx, ()> {
        debug!("leak_check(placeholders={:?})", placeholder_map);

        assert!(self.in_snapshot());

        // If the user gave `-Zno-leak-check`, then skip the leak
        // check completely. This is wildly unsound and also not
        // unlikely to cause an ICE or two. It is intended for use
        // only during a transition period, in which the MIR typeck
        // uses the "universe-style" check, and the rest of typeck
        // uses the more conservative leak check.  Since the leak
        // check is more conservative, we can't test the
        // universe-style check without disabling it.
        if tcx.sess.opts.debugging_opts.no_leak_check {
            return Ok(());
        }

        // Go through each placeholder that we created.
        for (_, &placeholder_region) in placeholder_map {
            // Find the universe this placeholder inhabits.
            let placeholder = match placeholder_region {
                ty::RePlaceholder(p) => p,
                _ => bug!(
                    "leak_check: expected placeholder found {:?}",
                    placeholder_region,
                ),
            };

            // Find all regions that are related to this placeholder
            // in some way. This means any region that either outlives
            // or is outlived by a placeholder.
            let mut taint_set = TaintSet::new(
                TaintDirections::both(),
                placeholder_region,
            );
            taint_set.fixed_point(tcx, &self.undo_log, &self.data.verifys);
            let tainted_regions = taint_set.into_set();

            // Report an error if two placeholders in the same universe
            // are related to one another, or if a placeholder is related
            // to something from a parent universe.
            for &tainted_region in &tainted_regions {
                if let ty::RePlaceholder(_) = tainted_region {
                    // Two placeholders cannot be related:
                    if tainted_region == placeholder_region {
                        continue;
                    }
                } else if self.universe(tainted_region).can_name(placeholder.universe) {
                    continue;
                }

                return Err(if overly_polymorphic {
                    debug!("Overly polymorphic!");
                    TypeError::RegionsOverlyPolymorphic(placeholder.name, tainted_region)
                } else {
                    debug!("Not as polymorphic!");
                    TypeError::RegionsInsufficientlyPolymorphic(placeholder.name, tainted_region)
                });
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
struct TaintSet<'tcx> {
    directions: TaintDirections,
    regions: FxHashSet<ty::Region<'tcx>>,
}

impl<'tcx> TaintSet<'tcx> {
    fn new(directions: TaintDirections, initial_region: ty::Region<'tcx>) -> Self {
        let mut regions = FxHashSet::default();
        regions.insert(initial_region);
        TaintSet {
            directions: directions,
            regions: regions,
        }
    }

    fn fixed_point(
        &mut self,
        tcx: TyCtxt<'tcx>,
        undo_log: &[UndoLog<'tcx>],
        verifys: &[Verify<'tcx>],
    ) {
        let mut prev_len = 0;
        while prev_len < self.len() {
            debug!(
                "tainted: prev_len = {:?} new_len = {:?}",
                prev_len,
                self.len()
            );

            prev_len = self.len();

            for undo_entry in undo_log {
                match undo_entry {
                    &AddConstraint(Constraint::VarSubVar(a, b)) => {
                        self.add_edge(tcx.mk_region(ReVar(a)), tcx.mk_region(ReVar(b)));
                    }
                    &AddConstraint(Constraint::RegSubVar(a, b)) => {
                        self.add_edge(a, tcx.mk_region(ReVar(b)));
                    }
                    &AddConstraint(Constraint::VarSubReg(a, b)) => {
                        self.add_edge(tcx.mk_region(ReVar(a)), b);
                    }
                    &AddConstraint(Constraint::RegSubReg(a, b)) => {
                        self.add_edge(a, b);
                    }
                    &AddGiven(a, b) => {
                        self.add_edge(a, tcx.mk_region(ReVar(b)));
                    }
                    &AddVerify(i) => span_bug!(
                        verifys[i].origin.span(),
                        "we never add verifications while doing higher-ranked things",
                    ),
                    &Purged | &AddCombination(..) | &AddVar(..) => {}
                }
            }
        }
    }

    fn into_set(self) -> FxHashSet<ty::Region<'tcx>> {
        self.regions
    }

    fn len(&self) -> usize {
        self.regions.len()
    }

    fn add_edge(&mut self, source: ty::Region<'tcx>, target: ty::Region<'tcx>) {
        if self.directions.incoming {
            if self.regions.contains(&target) {
                self.regions.insert(source);
            }
        }

        if self.directions.outgoing {
            if self.regions.contains(&source) {
                self.regions.insert(target);
            }
        }
    }
}
