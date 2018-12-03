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
use borrow_check::error_reporting::UseSpans;
use borrow_check::nll::ConstraintDescription;
use borrow_check::nll::region_infer::{Cause, RegionName};
use borrow_check::{Context, MirBorrowckCtxt, WriteKind};
use rustc::ty::{self, TyCtxt};
use rustc::mir::{
    CastKind, ConstraintCategory, FakeReadCause, Local, Location, Mir, Operand,
    Place, Projection, ProjectionElem, Rvalue, Statement, StatementKind,
    TerminatorKind
};
use rustc_errors::DiagnosticBuilder;
use syntax_pos::Span;

mod find_use;

pub(in borrow_check) enum BorrowExplanation {
    UsedLater(LaterUseKind, Span),
    UsedLaterInLoop(LaterUseKind, Span),
    UsedLaterWhenDropped {
        drop_loc: Location,
        dropped_local: Local,
        should_note_order: bool,
    },
    MustBeValidFor {
        category: ConstraintCategory,
        from_closure: bool,
        span: Span,
        region_name: RegionName,
        opt_place_desc: Option<String>,
    },
    Unexplained,
}

#[derive(Clone, Copy)]
pub(in borrow_check) enum LaterUseKind {
    TraitCapture,
    ClosureCapture,
    Call,
    FakeLetRead,
    Other,
}

impl BorrowExplanation {
    pub(in borrow_check) fn add_explanation_to_diagnostic<'cx, 'gcx, 'tcx>(
        &self,
        tcx: TyCtxt<'cx, 'gcx, 'tcx>,
        mir: &Mir<'tcx>,
        err: &mut DiagnosticBuilder<'_>,
        borrow_desc: &str,
    ) {
        match *self {
            BorrowExplanation::UsedLater(later_use_kind, var_or_use_span) => {
                let message = match later_use_kind {
                    LaterUseKind::TraitCapture => "borrow later captured here by trait object",
                    LaterUseKind::ClosureCapture => "borrow later captured here by closure",
                    LaterUseKind::Call =>  "borrow later used by call",
                    LaterUseKind::FakeLetRead => "borrow later stored here",
                    LaterUseKind::Other => "borrow later used here",
                };
                err.span_label(var_or_use_span, format!("{}{}", borrow_desc, message));
            },
            BorrowExplanation::UsedLaterInLoop(later_use_kind, var_or_use_span) => {
                let message = match later_use_kind {
                    LaterUseKind::TraitCapture =>
                        "borrow captured here by trait object, in later iteration of loop",
                    LaterUseKind::ClosureCapture =>
                        "borrow captured here by closure, in later iteration of loop",
                    LaterUseKind::Call =>  "borrow used by call, in later iteration of loop",
                    LaterUseKind::FakeLetRead => "borrow later stored here",
                    LaterUseKind::Other => "borrow used here, in later iteration of loop",
                };
                err.span_label(var_or_use_span, format!("{}{}", borrow_desc, message));
            },
            BorrowExplanation::UsedLaterWhenDropped { drop_loc, dropped_local,
                                                      should_note_order } =>
            {
                let local_decl = &mir.local_decls[dropped_local];
                let (dtor_desc, type_desc) = match local_decl.ty.sty {
                    // If type is an ADT that implements Drop, then
                    // simplify output by reporting just the ADT name.
                    ty::Adt(adt, _substs) if adt.has_dtor(tcx) && !adt.is_box() =>
                        ("`Drop` code", format!("type `{}`", tcx.item_path_str(adt.did))),

                    // Otherwise, just report the whole type (and use
                    // the intentionally fuzzy phrase "destructor")
                    ty::Closure(..) =>
                        ("destructor", "closure".to_owned()),
                    ty::Generator(..) =>
                        ("destructor", "generator".to_owned()),

                    _ => ("destructor", format!("type `{}`", local_decl.ty)),
                };

                match local_decl.name {
                    Some(local_name) => {
                        let message =
                            format!("{B}borrow might be used here, when `{LOC}` is dropped \
                                     and runs the {DTOR} for {TYPE}",
                                    B=borrow_desc, LOC=local_name, TYPE=type_desc, DTOR=dtor_desc);
                        err.span_label(mir.source_info(drop_loc).span, message);

                        if should_note_order {
                            err.note(
                                "values in a scope are dropped \
                                 in the opposite order they are defined",
                            );
                        }
                    }
                    None => {
                        err.span_label(local_decl.source_info.span,
                                       format!("a temporary with access to the {B}borrow \
                                                is created here ...",
                                               B=borrow_desc));
                        let message =
                            format!("... and the {B}borrow might be used here, \
                                     when that temporary is dropped \
                                     and runs the {DTOR} for {TYPE}",
                                    B=borrow_desc, TYPE=type_desc, DTOR=dtor_desc);
                        err.span_label(mir.source_info(drop_loc).span, message);

                        if let Some(info) = &local_decl.is_block_tail {
                            // FIXME: use span_suggestion instead, highlighting the
                            // whole block tail expression.
                            let msg = if info.tail_result_is_ignored {
                                "The temporary is part of an expression at the end of a block. \
                                 Consider adding semicolon after the expression so its temporaries \
                                 are dropped sooner, before the local variables declared by the \
                                 block are dropped."
                            } else {
                                "The temporary is part of an expression at the end of a block. \
                                 Consider forcing this temporary to be dropped sooner, before \
                                 the block's local variables are dropped. \
                                 For example, you could save the expression's value in a new \
                                 local variable `x` and then make `x` be the expression \
                                 at the end of the block."
                            };

                            err.note(msg);
                        }
                    }
                }
            },
            BorrowExplanation::MustBeValidFor {
                category,
                span,
                ref region_name,
                ref opt_place_desc,
                from_closure: _,
            } => {
                region_name.highlight_region_name(err);

                if let Some(desc) = opt_place_desc {
                    err.span_label(span, format!(
                        "{}requires that `{}` is borrowed for `{}`",
                        category.description(), desc, region_name,
                    ));
                } else {
                    err.span_label(span, format!(
                        "{}requires that {}borrow lasts for `{}`",
                        category.description(), borrow_desc, region_name,
                    ));
                };
            },
            _ => {},
        }
    }
}

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    /// Returns structured explanation for *why* the borrow contains the
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
    ///
    /// [d]: https://rust-lang.github.io/rfcs/2094-nll.html#leveraging-intuition-framing-errors-in-terms-of-points
    pub(in borrow_check) fn explain_why_borrow_contains_point(
        &self,
        context: Context,
        borrow: &BorrowData<'tcx>,
        kind_place: Option<(WriteKind, &Place<'tcx>)>,
    ) -> BorrowExplanation {
        debug!(
            "explain_why_borrow_contains_point(context={:?}, borrow={:?}, kind_place={:?})",
            context, borrow, kind_place
        );

        let regioncx = &self.nonlexical_regioncx;
        let mir = self.mir;
        let tcx = self.infcx.tcx;

        let borrow_region_vid = borrow.region;
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
            Some(Cause::LiveVar(local, location)) => {
                let span = mir.source_info(location).span;
                let spans = self.move_spans(&Place::Local(local), location)
                    .or_else(|| self.borrow_spans(span, location));

                if self.is_borrow_location_in_loop(context.loc) {
                    let later_use = self.later_use_kind(borrow, spans, location);
                    BorrowExplanation::UsedLaterInLoop(later_use.0, later_use.1)
                } else {
                    // Check if the location represents a `FakeRead`, and adapt the error
                    // message to the `FakeReadCause` it is from: in particular,
                    // the ones inserted in optimized `let var = <expr>` patterns.
                    let later_use = self.later_use_kind(borrow, spans, location);
                    BorrowExplanation::UsedLater(later_use.0, later_use.1)
                }
            }

             Some(Cause::DropVar(local, location)) => {
                 let mut should_note_order = false;
                 if mir.local_decls[local].name.is_some() {
                     if let Some((WriteKind::StorageDeadOrDrop, place)) = kind_place {
                         if let Place::Local(borrowed_local) = place {
                             let dropped_local_scope = mir.local_decls[local].visibility_scope;
                             let borrowed_local_scope =
                                 mir.local_decls[*borrowed_local].visibility_scope;

                             if mir.is_sub_scope(borrowed_local_scope, dropped_local_scope)
                                 && local != *borrowed_local
                             {
                                 should_note_order = true;
                             }
                         }
                     }
                 }

                 BorrowExplanation::UsedLaterWhenDropped {
                     drop_loc: location,
                     dropped_local: local,
                     should_note_order,
                 }
            }

            None => if let Some(region) = regioncx.to_error_region_vid(borrow_region_vid) {
                let (category, from_closure, span, region_name) = self
                    .nonlexical_regioncx
                    .free_region_constraint_info(
                        self.mir,
                        self.mir_def_id,
                        self.infcx,
                        borrow_region_vid,
                        region,
                    );
                let opt_place_desc = self.describe_place(&borrow.borrowed_place);
                BorrowExplanation::MustBeValidFor {
                    category,
                    from_closure,
                    span,
                    region_name,
                    opt_place_desc,
                }
            } else {
                BorrowExplanation::Unexplained
            }
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
                        pending_locations.extend(
                            targets.into_iter().map(|target| target.start_location()));
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
                        pending_locations.extend(
                            imaginary_targets.into_iter().map(|target| target.start_location()));
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

    /// Determine how the borrow was later used.
    fn later_use_kind(
        &self,
        borrow: &BorrowData<'tcx>,
        use_spans: UseSpans,
        location: Location
    ) -> (LaterUseKind, Span) {
        match use_spans {
            UseSpans::ClosureUse { var_span, .. } => {
                // Used in a closure.
                (LaterUseKind::ClosureCapture, var_span)
            },
            UseSpans::OtherUse(span) => {
                let block = &self.mir.basic_blocks()[location.block];

                let kind = if let Some(&Statement {
                    kind: StatementKind::FakeRead(FakeReadCause::ForLet, _),
                    ..
                }) = block.statements.get(location.statement_index) {
                    LaterUseKind::FakeLetRead
                } else if self.was_captured_by_trait_object(borrow) {
                    LaterUseKind::TraitCapture
                } else if location.statement_index == block.statements.len() {
                    if let TerminatorKind::Call {
                        ref func, from_hir_call: true, ..
                    } = block.terminator().kind {
                        // Just point to the function, to reduce the chance of overlapping spans.
                        let function_span = match func {
                            Operand::Constant(c) => c.span,
                            Operand::Copy(Place::Local(l)) | Operand::Move(Place::Local(l)) => {
                                let local_decl = &self.mir.local_decls[*l];
                                if local_decl.name.is_none() {
                                    local_decl.source_info.span
                                } else {
                                    span
                                }
                            },
                            _ => span,
                        };
                        return (LaterUseKind::Call, function_span);
                    } else {
                        LaterUseKind::Other
                    }
                } else {
                    LaterUseKind::Other
                };

                (kind, span)
            }
        }
    }

    /// Check if a borrowed value was captured by a trait object. We do this by
    /// looking forward in the MIR from the reserve location and checking if we see
    /// a unsized cast to a trait object on our data.
    fn was_captured_by_trait_object(&self, borrow: &BorrowData<'tcx>) -> bool {
        // Start at the reserve location, find the place that we want to see cast to a trait object.
        let location = borrow.reserve_location;
        let block = &self.mir[location.block];
        let stmt = block.statements.get(location.statement_index);
        debug!("was_captured_by_trait_object: location={:?} stmt={:?}", location, stmt);

        // We make a `queue` vector that has the locations we want to visit. As of writing, this
        // will only ever have one item at any given time, but by using a vector, we can pop from
        // it which simplifies the termination logic.
        let mut queue = vec![location];
        let mut target = if let Some(&Statement {
            kind: StatementKind::Assign(Place::Local(local), _),
            ..
        }) = stmt {
            local
        } else {
            return false;
        };

        debug!("was_captured_by_trait: target={:?} queue={:?}", target, queue);
        while let Some(current_location) = queue.pop() {
            debug!("was_captured_by_trait: target={:?}", target);
            let block = &self.mir[current_location.block];
            // We need to check the current location to find out if it is a terminator.
            let is_terminator = current_location.statement_index == block.statements.len();
            if !is_terminator {
                let stmt = &block.statements[current_location.statement_index];
                debug!("was_captured_by_trait_object: stmt={:?}", stmt);

                // The only kind of statement that we care about is assignments...
                if let StatementKind::Assign(
                    place,
                    box rvalue,
                ) = &stmt.kind {
                    let into = match place {
                        Place::Local(into) => into,
                        Place::Projection(box Projection {
                            base: Place::Local(into),
                            elem: ProjectionElem::Deref,
                        }) => into,
                        _ =>  {
                            // Continue at the next location.
                            queue.push(current_location.successor_within_block());
                            continue;
                        },
                    };

                    match rvalue {
                        // If we see a use, we should check whether it is our data, and if so
                        // update the place that we're looking for to that new place.
                        Rvalue::Use(operand) => match operand {
                            Operand::Copy(Place::Local(from)) |
                            Operand::Move(Place::Local(from)) if *from == target => {
                                target = *into;
                            },
                            _ => {},
                        },
                        // If we see a unsized cast, then if it is our data we should check
                        // whether it is being cast to a trait object.
                        Rvalue::Cast(CastKind::Unsize, operand, ty) => match operand {
                            Operand::Copy(Place::Local(from)) |
                            Operand::Move(Place::Local(from)) if *from == target => {
                                debug!("was_captured_by_trait_object: ty={:?}", ty);
                                // Check the type for a trait object.
                                return match ty.sty {
                                    // `&dyn Trait`
                                    ty::TyKind::Ref(_, ty, _) if ty.is_trait() => true,
                                    // `Box<dyn Trait>`
                                    _ if ty.is_box() && ty.boxed_ty().is_trait() =>
                                        true,
                                    // `dyn Trait`
                                    _ if ty.is_trait() => true,
                                    // Anything else.
                                    _ => false,
                                };
                            },
                            _ => return false,
                        },
                        _ => {},
                    }
                }

                // Continue at the next location.
                queue.push(current_location.successor_within_block());
            } else {
                // The only thing we need to do for terminators is progress to the next block.
                let terminator = block.terminator();
                debug!("was_captured_by_trait_object: terminator={:?}", terminator);

                if let TerminatorKind::Call {
                    destination: Some((Place::Local(dest), block)),
                    args,
                    ..
                } = &terminator.kind {
                    debug!(
                        "was_captured_by_trait_object: target={:?} dest={:?} args={:?}",
                        target, dest, args
                    );
                    // Check if one of the arguments to this function is the target place.
                    let found_target = args.iter().any(|arg| {
                        if let Operand::Move(Place::Local(potential)) = arg {
                            *potential == target
                        } else {
                            false
                        }
                    });

                    // If it is, follow this to the next block and update the target.
                    if found_target {
                        target = *dest;
                        queue.push(block.start_location());
                    }
                }
            }

            debug!("was_captured_by_trait: queue={:?}", queue);
        }

        // We didn't find anything and ran out of locations to check.
        false
    }
}
