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
use borrow_check::nll::region_infer::{Cause, RegionInferenceContext};
use borrow_check::{Context, MirBorrowckCtxt, WriteKind};
use rustc::mir::visit::{MirVisitable, PlaceContext, Visitor};
use rustc::mir::{Local, Location, Mir, Place};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::DiagnosticBuilder;
use util::liveness::{self, DefUse, LivenessMode};

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    /// Adds annotations to `err` explaining *why* the borrow contains the
    /// point from `context`. This is key for the "3-point errors"
    /// [described in the NLL RFC][d].
    ///
    /// # Parameters
    ///
    /// - `borrow`: the borrow in question
    /// - `context`: where the borrow occurs
    /// - `kind_place`: if Some, this describes the statement that triggered the error.
    ///   - first half is the kind of write, if any, being performed
    ///   - second half is the place being accessed
    /// - `err`: where the error annotations are going to be added
    ///
    /// [d]: https://rust-lang.github.io/rfcs/2094-nll.html#leveraging-intuition-framing-errors-in-terms-of-points
    pub(in borrow_check) fn explain_why_borrow_contains_point(
        &mut self,
        context: Context,
        borrow: &BorrowData<'tcx>,
        kind_place: Option<(WriteKind, &Place<'tcx>)>,
        err: &mut DiagnosticBuilder<'_>,
    ) {
        let regioncx = &&self.nonlexical_regioncx;
        let mir = self.mir;

        let borrow_region_vid = regioncx.to_region_vid(borrow.region);
        if let Some(cause) = regioncx.why_region_contains_point(borrow_region_vid, context.loc) {
            match cause {
                Cause::LiveVar(local, location) => match find_regular_use(
                    mir, regioncx, borrow, location, local,
                ) {
                    Some(p) => {
                        err.span_label(mir.source_info(p).span, format!("borrow later used here"));
                    }

                    None => {
                        span_bug!(
                            mir.source_info(context.loc).span,
                            "Cause should end in a LiveVar"
                        );
                    }
                },

                Cause::DropVar(local, location) => match find_drop_use(
                    mir, regioncx, borrow, location, local,
                ) {
                    Some(p) => match &mir.local_decls[local].name {
                        Some(local_name) => {
                            err.span_label(
                                mir.source_info(p).span,
                                format!("borrow later used here, when `{}` is dropped", local_name),
                            );

                            if let Some((WriteKind::StorageDeadOrDrop, place)) = kind_place {
                                if let Place::Local(borrowed_local) = place {
                                    let dropped_local_scope =
                                        mir.local_decls[local].visibility_scope;
                                    let borrowed_local_scope =
                                        mir.local_decls[*borrowed_local].visibility_scope;

                                    if mir.is_sub_scope(borrowed_local_scope, dropped_local_scope) {
                                        err.note(
                                            "values in a scope are dropped \
                                             in the opposite order they are defined",
                                        );
                                    }
                                }
                            }
                        }
                        None => {
                            err.span_label(
                                mir.local_decls[local].source_info.span,
                                "borrow may end up in a temporary, created here",
                            );

                            err.span_label(
                                mir.source_info(p).span,
                                "temporary later dropped here, \
                                 potentially using the reference",
                            );
                        }
                    },

                    None => {
                        span_bug!(
                            mir.source_info(context.loc).span,
                            "Cause should end in a DropVar"
                        );
                    }
                },

                Cause::UniversalRegion(region_vid) => {
                    if let Some(region) = regioncx.to_error_region(region_vid) {
                        self.tcx.note_and_explain_free_region(
                            err,
                            "borrowed value must be valid for ",
                            region,
                            "...",
                        );
                    }
                }

                _ => {}
            }
        }
    }
}

fn find_regular_use<'gcx, 'tcx>(
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

fn find_drop_use<'gcx, 'tcx>(
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
