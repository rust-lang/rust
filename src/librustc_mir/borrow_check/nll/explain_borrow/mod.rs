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
use borrow_check::nll::region_infer::Cause;
use borrow_check::{Context, MirBorrowckCtxt, WriteKind};
use rustc::mir::{FakeReadCause, Local, Location, Place, StatementKind, TerminatorKind};
use rustc_errors::DiagnosticBuilder;
use rustc::ty::Region;

mod find_use;

#[derive(Copy, Clone, Debug)]
pub enum BorrowContainsPointReason<'tcx> {
    Liveness {
        local: Local,
        location: Location,
        in_loop: bool,
    },
    DropLiveness {
        local: Local,
        location: Location,
    },
    OutlivesFreeRegion {
        outlived_region: Option<Region<'tcx>>,
    },
}

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
        &self,
        context: Context,
        borrow: &BorrowData<'tcx>,
        kind_place: Option<(WriteKind, &Place<'tcx>)>,
        err: &mut DiagnosticBuilder<'_>,
    ) {
        let reason = self.find_why_borrow_contains_point(context, borrow);
        self.report_why_borrow_contains_point(err, reason, kind_place);
    }

    /// Finds the reason that [explain_why_borrow_contains_point] will report
    /// but doesn't add it to any message. This is a separate function in case
    /// the caller wants to change the error they report based on the reason
    /// that will be reported.
    pub(in borrow_check) fn find_why_borrow_contains_point(
        &self,
        context: Context,
        borrow: &BorrowData<'tcx>
    ) -> BorrowContainsPointReason<'tcx> {
        use self::BorrowContainsPointReason::*;

        debug!(
            "find_why_borrow_contains_point(context={:?}, borrow={:?})",
            context, borrow,
        );

        let regioncx = &self.nonlexical_regioncx;
        let mir = self.mir;
        let tcx = self.tcx;

        let borrow_region_vid = regioncx.to_region_vid(borrow.region);

        debug!(
            "explain_why_borrow_contains_point: borrow_region_vid={:?}",
            borrow_region_vid
        );

        let region_sub = regioncx.find_sub_region_live_at(borrow_region_vid, context.loc);

        debug!(
            "explain_why_borrow_contains_point: region_sub={:?}",
            region_sub
        );

        match find_use::find(mir, regioncx, tcx, region_sub, context.loc) {
            Some(Cause::LiveVar(local, location)) => Liveness {
                local,
                location,
                in_loop: self.is_borrow_location_in_loop(context.loc),
            },
            Some(Cause::DropVar(local, location)) => DropLiveness {
                local,
                location,
            },
            None => OutlivesFreeRegion {
                outlived_region: regioncx.to_error_region(region_sub),
            },
        }
    }

    /// Adds annotations to `err` for the explanation `reason`. This is a
    /// separate method so that the caller can change their error message based
    /// on the reason that is going to be reported.
    pub (in borrow_check) fn report_why_borrow_contains_point(
        &self,
        err: &mut DiagnosticBuilder,
        reason: BorrowContainsPointReason<'tcx>,
        kind_place: Option<(WriteKind, &Place<'tcx>)>,
    ) {
        use self::BorrowContainsPointReason::*;

        debug!(
            "find_why_borrow_contains_point(reason={:?}, kind_place={:?})",
            reason, kind_place,
        );

        let mir = self.mir;

        match reason {
            Liveness { local, location, in_loop } => {
                let span = mir.source_info(location).span;
                let spans = self.move_spans(&Place::Local(local), location)
                    .or_else(|| self.borrow_spans(span, location));
                let message = if in_loop {
                    if spans.for_closure() {
                        "borrow captured here by closure in later iteration of loop"
                    } else {
                        "borrow used here in later iteration of loop"
                    }
                } else {
                    if spans.for_closure() {
                        "borrow later captured here by closure"
                    } else {
                        // Check if the location represents a `FakeRead`, and adapt the error
                        // message to the `FakeReadCause` it is from: in particular,
                        // the ones inserted in optimized `let var = <expr>` patterns.
                        let is_fake_read_for_let = match self.mir.basic_blocks()[location.block]
                            .statements
                            .get(location.statement_index)
                        {
                            None => false,
                            Some(stmt) => {
                                if let StatementKind::FakeRead(ref cause, _) = stmt.kind {
                                    match cause {
                                        FakeReadCause::ForLet => true,
                                        _ => false,
                                    }
                                } else {
                                    false
                                }
                            }
                        };

                        if is_fake_read_for_let {
                            "borrow later stored here"
                        } else {
                            "borrow later used here"
                        }
                    }
                };
                err.span_label(spans.var_or_use(), message);
            }
            DropLiveness { local, location } => match &mir.local_decls[local].name {
                Some(local_name) => {
                    err.span_label(
                        mir.source_info(location).span,
                        format!("borrow later used here, when `{}` is dropped", local_name),
                    );

                    if let Some((WriteKind::StorageDeadOrDrop, place)) = kind_place {
                        if let Place::Local(borrowed_local) = place {
                            let dropped_local_scope = mir.local_decls[local].visibility_scope;
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

                None => {}
            }
            OutlivesFreeRegion { outlived_region: Some(region) } => {
                self.tcx.note_and_explain_free_region(
                    err,
                    "borrowed value must be valid for ",
                    region,
                    "...",
                );
            }
            OutlivesFreeRegion { outlived_region: None } => (),
        }
    }

    /// Check if a borrow location is within a loop.
    fn is_borrow_location_in_loop(
        &self,
        borrow_location: Location,
    ) -> bool {
        let mut visited_locations = Vec::new();
        let mut pending_locations = vec![ borrow_location ];
        debug!("is_in_loop: borrow_location={:?}", borrow_location);

        while let Some(location) = pending_locations.pop() {
            debug!("is_in_loop: location={:?} pending_locations={:?} visited_locations={:?}",
                   location, pending_locations, visited_locations);
            if location == borrow_location && visited_locations.contains(&borrow_location) {
                // We've managed to return to where we started (and this isn't the start of the
                // search).
                debug!("is_in_loop: found!");
                return true;
            }

            // Skip locations we've been.
            if visited_locations.contains(&location) { continue; }

            let block = &self.mir.basic_blocks()[location.block];
            if location.statement_index ==  block.statements.len() {
                // Add start location of the next blocks to pending locations.
                match block.terminator().kind {
                    TerminatorKind::Goto { target } => {
                        pending_locations.push(target.start_location());
                    },
                    TerminatorKind::SwitchInt { ref targets, .. } => {
                        for target in targets {
                            pending_locations.push(target.start_location());
                        }
                    },
                    TerminatorKind::Drop { target, unwind, .. } |
                    TerminatorKind::DropAndReplace { target, unwind, .. } |
                    TerminatorKind::Assert { target, cleanup: unwind, .. } |
                    TerminatorKind::Yield { resume: target, drop: unwind, .. } |
                    TerminatorKind::FalseUnwind { real_target: target, unwind, .. } => {
                        pending_locations.push(target.start_location());
                        if let Some(unwind) = unwind {
                            pending_locations.push(unwind.start_location());
                        }
                    },
                    TerminatorKind::Call { ref destination, cleanup, .. } => {
                        if let Some((_, destination)) = destination {
                            pending_locations.push(destination.start_location());
                        }
                        if let Some(cleanup) = cleanup {
                            pending_locations.push(cleanup.start_location());
                        }
                    },
                    TerminatorKind::FalseEdges { real_target, ref imaginary_targets, .. } => {
                        pending_locations.push(real_target.start_location());
                        for target in imaginary_targets {
                            pending_locations.push(target.start_location());
                        }
                    },
                    _ => {},
                }
            } else {
                // Add the next statement to pending locations.
                pending_locations.push(location.successor_within_block());
            }

            // Keep track of where we have visited.
            visited_locations.push(location);
        }

        false
    }
}

