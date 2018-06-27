// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::borrow_set::BorrowData;
use borrow_check::nll::region_infer::RegionInferenceContext;
use rustc::mir::visit::{MirVisitable, PlaceContext, Visitor};
use rustc::mir::{Local, Location, Mir};
use rustc_data_structures::fx::FxHashSet;
use util::liveness::{self, DefUse, LivenessMode};

crate fn regular_use<'gcx, 'tcx>(
    mir: &'gcx Mir,
    regioncx: &'tcx RegionInferenceContext,
    borrow: &'tcx BorrowData,
    start_point: Location,
    local: Local,
) -> Option<Location> {
    let mut uf = UseFinder {
        mir,
        regioncx,
        borrow,
        start_point,
        local,
        liveness_mode: LivenessMode {
            include_regular_use: true,
            include_drops: false,
        },
    };

    uf.find()
}

crate fn drop_use<'gcx, 'tcx>(
    mir: &'gcx Mir,
    regioncx: &'tcx RegionInferenceContext,
    borrow: &'tcx BorrowData,
    start_point: Location,
    local: Local,
) -> Option<Location> {
    let mut uf = UseFinder {
        mir,
        regioncx,
        borrow,
        start_point,
        local,
        liveness_mode: LivenessMode {
            include_regular_use: false,
            include_drops: true,
        },
    };

    uf.find()
}

struct UseFinder<'gcx, 'tcx> {
    mir: &'gcx Mir<'gcx>,
    regioncx: &'tcx RegionInferenceContext<'tcx>,
    borrow: &'tcx BorrowData<'tcx>,
    start_point: Location,
    local: Local,
    liveness_mode: LivenessMode,
}

impl<'gcx, 'tcx> UseFinder<'gcx, 'tcx> {
    fn find(&mut self) -> Option<Location> {
        let mut stack = vec![];
        let mut visited = FxHashSet();

        stack.push(self.start_point);
        while let Some(p) = stack.pop() {
            if !self.regioncx.region_contains_point(self.borrow.region, p) {
                continue;
            }

            if !visited.insert(p) {
                continue;
            }

            let block_data = &self.mir[p.block];
            let (defined, used) = self.def_use(p, block_data.visitable(p.statement_index));

            if used {
                return Some(p);
            } else if !defined {
                if p.statement_index < block_data.statements.len() {
                    stack.push(Location {
                        statement_index: p.statement_index + 1,
                        ..p
                    });
                } else {
                    stack.extend(block_data.terminator().successors().map(|&basic_block| {
                        Location {
                            statement_index: 0,
                            block: basic_block,
                        }
                    }));
                }
            }
        }

        None
    }

    fn def_use(&self, location: Location, thing: &dyn MirVisitable<'tcx>) -> (bool, bool) {
        let mut visitor = DefUseVisitor {
            defined: false,
            used: false,
            local: self.local,
            liveness_mode: self.liveness_mode,
        };

        thing.apply(location, &mut visitor);

        (visitor.defined, visitor.used)
    }
}

struct DefUseVisitor {
    defined: bool,
    used: bool,
    local: Local,
    liveness_mode: LivenessMode,
}

impl<'tcx> Visitor<'tcx> for DefUseVisitor {
    fn visit_local(&mut self, &local: &Local, context: PlaceContext<'tcx>, _: Location) {
        if local == self.local {
            match liveness::categorize(context, self.liveness_mode) {
                Some(DefUse::Def) => self.defined = true,
                Some(DefUse::Use) => self.used = true,
                None => (),
            }
        }
    }
}
