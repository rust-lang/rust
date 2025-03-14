use std::collections::VecDeque;

use rustc_data_structures::fx::FxIndexSet;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::{self, Body, Local, Location};
use rustc_middle::ty::{RegionVid, TyCtxt};

use crate::def_use::{self, DefUse};
use crate::region_infer::{Cause, RegionInferenceContext};

pub(crate) fn find<'tcx>(
    body: &Body<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    tcx: TyCtxt<'tcx>,
    region_vid: RegionVid,
    start_point: Location,
) -> Option<Cause> {
    let mut uf = UseFinder { body, regioncx, tcx, region_vid, start_point };

    uf.find()
}

struct UseFinder<'a, 'tcx> {
    body: &'a Body<'tcx>,
    regioncx: &'a RegionInferenceContext<'tcx>,
    tcx: TyCtxt<'tcx>,
    region_vid: RegionVid,
    start_point: Location,
}

impl<'a, 'tcx> UseFinder<'a, 'tcx> {
    fn find(&mut self) -> Option<Cause> {
        let mut queue = VecDeque::new();
        let mut visited = FxIndexSet::default();

        queue.push_back(self.start_point);
        while let Some(p) = queue.pop_front() {
            if !self.regioncx.region_contains(self.region_vid, p) {
                continue;
            }

            if !visited.insert(p) {
                continue;
            }

            let block_data = &self.body[p.block];

            let mut visitor = DefUseVisitor {
                body: self.body,
                tcx: self.tcx,
                region_vid: self.region_vid,
                def_use_result: None,
            };

            let is_statement = p.statement_index < block_data.statements.len();

            if is_statement {
                visitor.visit_statement(&block_data.statements[p.statement_index], p);
            } else {
                visitor.visit_terminator(block_data.terminator.as_ref().unwrap(), p);
            }

            match visitor.def_use_result {
                Some(DefUseResult::Def) => {}

                Some(DefUseResult::UseLive { local }) => {
                    return Some(Cause::LiveVar(local, p));
                }

                Some(DefUseResult::UseDrop { local }) => {
                    return Some(Cause::DropVar(local, p));
                }

                None => {
                    if is_statement {
                        queue.push_back(p.successor_within_block());
                    } else {
                        queue.extend(
                            block_data
                                .terminator()
                                .successors()
                                .filter(|&bb| {
                                    Some(&mir::UnwindAction::Cleanup(bb))
                                        != block_data.terminator().unwind()
                                })
                                .map(|bb| Location { statement_index: 0, block: bb }),
                        );
                    }
                }
            }
        }

        None
    }
}

struct DefUseVisitor<'a, 'tcx> {
    body: &'a Body<'tcx>,
    tcx: TyCtxt<'tcx>,
    region_vid: RegionVid,
    def_use_result: Option<DefUseResult>,
}

enum DefUseResult {
    Def,
    UseLive { local: Local },
    UseDrop { local: Local },
}

impl<'a, 'tcx> Visitor<'tcx> for DefUseVisitor<'a, 'tcx> {
    fn visit_local(&mut self, local: Local, context: PlaceContext, _: Location) {
        let local_ty = self.body.local_decls[local].ty;

        let mut found_it = false;
        self.tcx.for_each_free_region(&local_ty, |r| {
            if r.as_var() == self.region_vid {
                found_it = true;
            }
        });

        if found_it {
            self.def_use_result = match def_use::categorize(context) {
                Some(DefUse::Def) => Some(DefUseResult::Def),
                Some(DefUse::Use) => Some(DefUseResult::UseLive { local }),
                Some(DefUse::Drop) => Some(DefUseResult::UseDrop { local }),
                None => None,
            };
        }
    }
}
