use super::*;

#[derive(Debug)]
pub(super) struct TaintSet<'tcx> {
    directions: TaintDirections,
    regions: FxHashSet<ty::Region<'tcx>>,
}

impl<'tcx> TaintSet<'tcx> {
    pub(super) fn new(directions: TaintDirections, initial_region: ty::Region<'tcx>) -> Self {
        let mut regions = FxHashSet::default();
        regions.insert(initial_region);
        TaintSet {
            directions: directions,
            regions: regions,
        }
    }

    pub(super) fn fixed_point(
        &mut self,
        tcx: TyCtxt<'_, '_, 'tcx>,
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
                    &AddVerify(i) => {
                        span_bug!(
                            verifys[i].origin.span(),
                            "we never add verifications while doing higher-ranked things",
                        )
                    }
                    &Purged | &AddCombination(..) | &AddVar(..) => {}
                }
            }
        }
    }

    pub(super) fn into_set(self) -> FxHashSet<ty::Region<'tcx>> {
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
