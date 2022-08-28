//! Print diagnostics to explain why values are borrowed.

use std::collections::VecDeque;

use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{Applicability, Diagnostic};
use rustc_index::vec::IndexVec;
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_middle::mir::{
    Body, CastKind, ConstraintCategory, FakeReadCause, Local, Location, Operand, Place, Rvalue,
    Statement, StatementKind, TerminatorKind,
};
use rustc_middle::ty::adjustment::PointerCast;
use rustc_middle::ty::{self, RegionVid, TyCtxt};
use rustc_span::symbol::{kw, Symbol};
use rustc_span::{sym, DesugaringKind, Span};

use crate::region_infer::BlameConstraint;
use crate::{
    borrow_set::BorrowData, nll::ConstraintDescription, region_infer::Cause, MirBorrowckCtxt,
    WriteKind,
};

use super::{find_use, RegionName, UseSpans};

#[derive(Debug)]
pub(crate) enum BorrowExplanation<'tcx> {
    UsedLater(LaterUseKind, Span, Option<Span>),
    UsedLaterInLoop(LaterUseKind, Span, Option<Span>),
    UsedLaterWhenDropped {
        drop_loc: Location,
        dropped_local: Local,
        should_note_order: bool,
    },
    MustBeValidFor {
        category: ConstraintCategory<'tcx>,
        from_closure: bool,
        span: Span,
        region_name: RegionName,
        opt_place_desc: Option<String>,
    },
    Unexplained,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum LaterUseKind {
    TraitCapture,
    ClosureCapture,
    Call,
    FakeLetRead,
    Other,
}

impl<'tcx> BorrowExplanation<'tcx> {
    pub(crate) fn is_explained(&self) -> bool {
        !matches!(self, BorrowExplanation::Unexplained)
    }
    pub(crate) fn add_explanation_to_diagnostic(
        &self,
        tcx: TyCtxt<'tcx>,
        body: &Body<'tcx>,
        local_names: &IndexVec<Local, Option<Symbol>>,
        err: &mut Diagnostic,
        borrow_desc: &str,
        borrow_span: Option<Span>,
        multiple_borrow_span: Option<(Span, Span)>,
    ) {
        match *self {
            BorrowExplanation::UsedLater(later_use_kind, var_or_use_span, path_span) => {
                let message = match later_use_kind {
                    LaterUseKind::TraitCapture => "captured here by trait object",
                    LaterUseKind::ClosureCapture => "captured here by closure",
                    LaterUseKind::Call => "used by call",
                    LaterUseKind::FakeLetRead => "stored here",
                    LaterUseKind::Other => "used here",
                };
                // We can use `var_or_use_span` if either `path_span` is not present, or both spans are the same
                if path_span.map(|path_span| path_span == var_or_use_span).unwrap_or(true) {
                    if borrow_span.map(|sp| !sp.overlaps(var_or_use_span)).unwrap_or(true) {
                        err.span_label(
                            var_or_use_span,
                            format!("{}borrow later {}", borrow_desc, message),
                        );
                    }
                } else {
                    // path_span must be `Some` as otherwise the if condition is true
                    let path_span = path_span.unwrap();
                    // path_span is only present in the case of closure capture
                    assert!(matches!(later_use_kind, LaterUseKind::ClosureCapture));
                    if !borrow_span.map_or(false, |sp| sp.overlaps(var_or_use_span)) {
                        let path_label = "used here by closure";
                        let capture_kind_label = message;
                        err.span_label(
                            var_or_use_span,
                            format!("{}borrow later {}", borrow_desc, capture_kind_label),
                        );
                        err.span_label(path_span, path_label);
                    }
                }
            }
            BorrowExplanation::UsedLaterInLoop(later_use_kind, var_or_use_span, path_span) => {
                let message = match later_use_kind {
                    LaterUseKind::TraitCapture => {
                        "borrow captured here by trait object, in later iteration of loop"
                    }
                    LaterUseKind::ClosureCapture => {
                        "borrow captured here by closure, in later iteration of loop"
                    }
                    LaterUseKind::Call => "borrow used by call, in later iteration of loop",
                    LaterUseKind::FakeLetRead => "borrow later stored here",
                    LaterUseKind::Other => "borrow used here, in later iteration of loop",
                };
                // We can use `var_or_use_span` if either `path_span` is not present, or both spans are the same
                if path_span.map(|path_span| path_span == var_or_use_span).unwrap_or(true) {
                    err.span_label(var_or_use_span, format!("{}{}", borrow_desc, message));
                } else {
                    // path_span must be `Some` as otherwise the if condition is true
                    let path_span = path_span.unwrap();
                    // path_span is only present in the case of closure capture
                    assert!(matches!(later_use_kind, LaterUseKind::ClosureCapture));
                    if borrow_span.map(|sp| !sp.overlaps(var_or_use_span)).unwrap_or(true) {
                        let path_label = "used here by closure";
                        let capture_kind_label = message;
                        err.span_label(
                            var_or_use_span,
                            format!("{}borrow later {}", borrow_desc, capture_kind_label),
                        );
                        err.span_label(path_span, path_label);
                    }
                }
            }
            BorrowExplanation::UsedLaterWhenDropped {
                drop_loc,
                dropped_local,
                should_note_order,
            } => {
                let local_decl = &body.local_decls[dropped_local];
                let mut ty = local_decl.ty;
                if local_decl.source_info.span.desugaring_kind() == Some(DesugaringKind::ForLoop) {
                    if let ty::Adt(adt, substs) = local_decl.ty.kind() {
                        if tcx.is_diagnostic_item(sym::Option, adt.did()) {
                            // in for loop desugaring, only look at the `Some(..)` inner type
                            ty = substs.type_at(0);
                        }
                    }
                }
                let (dtor_desc, type_desc) = match ty.kind() {
                    // If type is an ADT that implements Drop, then
                    // simplify output by reporting just the ADT name.
                    ty::Adt(adt, _substs) if adt.has_dtor(tcx) && !adt.is_box() => {
                        ("`Drop` code", format!("type `{}`", tcx.def_path_str(adt.did())))
                    }

                    // Otherwise, just report the whole type (and use
                    // the intentionally fuzzy phrase "destructor")
                    ty::Closure(..) => ("destructor", "closure".to_owned()),
                    ty::Generator(..) => ("destructor", "generator".to_owned()),

                    _ => ("destructor", format!("type `{}`", local_decl.ty)),
                };

                match local_names[dropped_local] {
                    Some(local_name) if !local_decl.from_compiler_desugaring() => {
                        let message = format!(
                            "{B}borrow might be used here, when `{LOC}` is dropped \
                             and runs the {DTOR} for {TYPE}",
                            B = borrow_desc,
                            LOC = local_name,
                            TYPE = type_desc,
                            DTOR = dtor_desc
                        );
                        err.span_label(body.source_info(drop_loc).span, message);

                        if should_note_order {
                            err.note(
                                "values in a scope are dropped \
                                 in the opposite order they are defined",
                            );
                        }
                    }
                    _ => {
                        err.span_label(
                            local_decl.source_info.span,
                            format!(
                                "a temporary with access to the {B}borrow \
                                 is created here ...",
                                B = borrow_desc
                            ),
                        );
                        let message = format!(
                            "... and the {B}borrow might be used here, \
                             when that temporary is dropped \
                             and runs the {DTOR} for {TYPE}",
                            B = borrow_desc,
                            TYPE = type_desc,
                            DTOR = dtor_desc
                        );
                        err.span_label(body.source_info(drop_loc).span, message);

                        if let Some(info) = &local_decl.is_block_tail {
                            if info.tail_result_is_ignored {
                                // #85581: If the first mutable borrow's scope contains
                                // the second borrow, this suggestion isn't helpful.
                                if !multiple_borrow_span
                                    .map(|(old, new)| {
                                        old.to(info.span.shrink_to_hi()).contains(new)
                                    })
                                    .unwrap_or(false)
                                {
                                    err.span_suggestion_verbose(
                                        info.span.shrink_to_hi(),
                                        "consider adding semicolon after the expression so its \
                                        temporaries are dropped sooner, before the local variables \
                                        declared by the block are dropped",
                                        ";",
                                        Applicability::MaybeIncorrect,
                                    );
                                }
                            } else {
                                err.note(
                                    "the temporary is part of an expression at the end of a \
                                     block;\nconsider forcing this temporary to be dropped sooner, \
                                     before the block's local variables are dropped",
                                );
                                err.multipart_suggestion(
                                    "for example, you could save the expression's value in a new \
                                     local variable `x` and then make `x` be the expression at the \
                                     end of the block",
                                    vec![
                                        (info.span.shrink_to_lo(), "let x = ".to_string()),
                                        (info.span.shrink_to_hi(), "; x".to_string()),
                                    ],
                                    Applicability::MaybeIncorrect,
                                );
                            };
                        }
                    }
                }
            }
            BorrowExplanation::MustBeValidFor {
                category,
                span,
                ref region_name,
                ref opt_place_desc,
                from_closure: _,
            } => {
                region_name.highlight_region_name(err);

                if let Some(desc) = opt_place_desc {
                    err.span_label(
                        span,
                        format!(
                            "{}requires that `{}` is borrowed for `{}`",
                            category.description(),
                            desc,
                            region_name,
                        ),
                    );
                } else {
                    err.span_label(
                        span,
                        format!(
                            "{}requires that {}borrow lasts for `{}`",
                            category.description(),
                            borrow_desc,
                            region_name,
                        ),
                    );
                };

                self.add_lifetime_bound_suggestion_to_diagnostic(err, &category, span, region_name);
            }
            _ => {}
        }
    }

    fn add_lifetime_bound_suggestion_to_diagnostic(
        &self,
        err: &mut Diagnostic,
        category: &ConstraintCategory<'tcx>,
        span: Span,
        region_name: &RegionName,
    ) {
        if !span.is_desugaring(DesugaringKind::OpaqueTy) {
            return;
        }
        if let ConstraintCategory::OpaqueType = category {
            let suggestable_name =
                if region_name.was_named() { region_name.name } else { kw::UnderscoreLifetime };

            let msg = format!(
                "you can add a bound to the {}to make it last less than `'static` and match `{}`",
                category.description(),
                region_name,
            );

            err.span_suggestion_verbose(
                span.shrink_to_hi(),
                &msg,
                format!(" + {}", suggestable_name),
                Applicability::Unspecified,
            );
        }
    }
}

impl<'cx, 'tcx> MirBorrowckCtxt<'cx, 'tcx> {
    fn free_region_constraint_info(
        &self,
        borrow_region: RegionVid,
        outlived_region: RegionVid,
    ) -> (ConstraintCategory<'tcx>, bool, Span, Option<RegionName>) {
        let BlameConstraint { category, from_closure, cause, variance_info: _ } =
            self.regioncx.best_blame_constraint(
                &self.body,
                borrow_region,
                NllRegionVariableOrigin::FreeRegion,
                |r| self.regioncx.provides_universal_region(r, borrow_region, outlived_region),
            );

        let outlived_fr_name = self.give_region_a_name(outlived_region);

        (category, from_closure, cause.span, outlived_fr_name)
    }

    /// Returns structured explanation for *why* the borrow contains the
    /// point from `location`. This is key for the "3-point errors"
    /// [described in the NLL RFC][d].
    ///
    /// # Parameters
    ///
    /// - `borrow`: the borrow in question
    /// - `location`: where the borrow occurs
    /// - `kind_place`: if Some, this describes the statement that triggered the error.
    ///   - first half is the kind of write, if any, being performed
    ///   - second half is the place being accessed
    ///
    /// [d]: https://rust-lang.github.io/rfcs/2094-nll.html#leveraging-intuition-framing-errors-in-terms-of-points
    pub(crate) fn explain_why_borrow_contains_point(
        &self,
        location: Location,
        borrow: &BorrowData<'tcx>,
        kind_place: Option<(WriteKind, Place<'tcx>)>,
    ) -> BorrowExplanation<'tcx> {
        debug!(
            "explain_why_borrow_contains_point(location={:?}, borrow={:?}, kind_place={:?})",
            location, borrow, kind_place
        );

        let regioncx = &self.regioncx;
        let body: &Body<'_> = &self.body;
        let tcx = self.infcx.tcx;

        let borrow_region_vid = borrow.region;
        debug!("explain_why_borrow_contains_point: borrow_region_vid={:?}", borrow_region_vid);

        let region_sub = self.regioncx.find_sub_region_live_at(borrow_region_vid, location);
        debug!("explain_why_borrow_contains_point: region_sub={:?}", region_sub);

        match find_use::find(body, regioncx, tcx, region_sub, location) {
            Some(Cause::LiveVar(local, location)) => {
                let span = body.source_info(location).span;
                let spans = self
                    .move_spans(Place::from(local).as_ref(), location)
                    .or_else(|| self.borrow_spans(span, location));

                let borrow_location = location;
                if self.is_use_in_later_iteration_of_loop(borrow_location, location) {
                    let later_use = self.later_use_kind(borrow, spans, location);
                    BorrowExplanation::UsedLaterInLoop(later_use.0, later_use.1, later_use.2)
                } else {
                    // Check if the location represents a `FakeRead`, and adapt the error
                    // message to the `FakeReadCause` it is from: in particular,
                    // the ones inserted in optimized `let var = <expr>` patterns.
                    let later_use = self.later_use_kind(borrow, spans, location);
                    BorrowExplanation::UsedLater(later_use.0, later_use.1, later_use.2)
                }
            }

            Some(Cause::DropVar(local, location)) => {
                let mut should_note_order = false;
                if self.local_names[local].is_some()
                    && let Some((WriteKind::StorageDeadOrDrop, place)) = kind_place
                    && let Some(borrowed_local) = place.as_local()
                    && self.local_names[borrowed_local].is_some() && local != borrowed_local
                {
                    should_note_order = true;
                }

                BorrowExplanation::UsedLaterWhenDropped {
                    drop_loc: location,
                    dropped_local: local,
                    should_note_order,
                }
            }

            None => {
                if let Some(region) = self.to_error_region_vid(borrow_region_vid) {
                    let (category, from_closure, span, region_name) =
                        self.free_region_constraint_info(borrow_region_vid, region);
                    if let Some(region_name) = region_name {
                        let opt_place_desc = self.describe_place(borrow.borrowed_place.as_ref());
                        BorrowExplanation::MustBeValidFor {
                            category,
                            from_closure,
                            span,
                            region_name,
                            opt_place_desc,
                        }
                    } else {
                        debug!(
                            "explain_why_borrow_contains_point: \
                             Could not generate a region name"
                        );
                        BorrowExplanation::Unexplained
                    }
                } else {
                    debug!(
                        "explain_why_borrow_contains_point: \
                         Could not generate an error region vid"
                    );
                    BorrowExplanation::Unexplained
                }
            }
        }
    }

    /// true if `borrow_location` can reach `use_location` by going through a loop and
    /// `use_location` is also inside of that loop
    fn is_use_in_later_iteration_of_loop(
        &self,
        borrow_location: Location,
        use_location: Location,
    ) -> bool {
        let back_edge = self.reach_through_backedge(borrow_location, use_location);
        back_edge.map_or(false, |back_edge| self.can_reach_head_of_loop(use_location, back_edge))
    }

    /// Returns the outmost back edge if `from` location can reach `to` location passing through
    /// that back edge
    fn reach_through_backedge(&self, from: Location, to: Location) -> Option<Location> {
        let mut visited_locations = FxHashSet::default();
        let mut pending_locations = VecDeque::new();
        visited_locations.insert(from);
        pending_locations.push_back(from);
        debug!("reach_through_backedge: from={:?} to={:?}", from, to,);

        let mut outmost_back_edge = None;
        while let Some(location) = pending_locations.pop_front() {
            debug!(
                "reach_through_backedge: location={:?} outmost_back_edge={:?}
                   pending_locations={:?} visited_locations={:?}",
                location, outmost_back_edge, pending_locations, visited_locations
            );

            if location == to && outmost_back_edge.is_some() {
                // We've managed to reach the use location
                debug!("reach_through_backedge: found!");
                return outmost_back_edge;
            }

            let block = &self.body.basic_blocks[location.block];

            if location.statement_index < block.statements.len() {
                let successor = location.successor_within_block();
                if visited_locations.insert(successor) {
                    pending_locations.push_back(successor);
                }
            } else {
                pending_locations.extend(
                    block
                        .terminator()
                        .successors()
                        .map(|bb| Location { statement_index: 0, block: bb })
                        .filter(|s| visited_locations.insert(*s))
                        .map(|s| {
                            if self.is_back_edge(location, s) {
                                match outmost_back_edge {
                                    None => {
                                        outmost_back_edge = Some(location);
                                    }

                                    Some(back_edge)
                                        if location.dominates(back_edge, &self.dominators) =>
                                    {
                                        outmost_back_edge = Some(location);
                                    }

                                    Some(_) => {}
                                }
                            }

                            s
                        }),
                );
            }
        }

        None
    }

    /// true if `from` location can reach `loop_head` location and `loop_head` dominates all the
    /// intermediate nodes
    fn can_reach_head_of_loop(&self, from: Location, loop_head: Location) -> bool {
        self.find_loop_head_dfs(from, loop_head, &mut FxHashSet::default())
    }

    fn find_loop_head_dfs(
        &self,
        from: Location,
        loop_head: Location,
        visited_locations: &mut FxHashSet<Location>,
    ) -> bool {
        visited_locations.insert(from);

        if from == loop_head {
            return true;
        }

        if loop_head.dominates(from, &self.dominators) {
            let block = &self.body.basic_blocks[from.block];

            if from.statement_index < block.statements.len() {
                let successor = from.successor_within_block();

                if !visited_locations.contains(&successor)
                    && self.find_loop_head_dfs(successor, loop_head, visited_locations)
                {
                    return true;
                }
            } else {
                for bb in block.terminator().successors() {
                    let successor = Location { statement_index: 0, block: bb };

                    if !visited_locations.contains(&successor)
                        && self.find_loop_head_dfs(successor, loop_head, visited_locations)
                    {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// True if an edge `source -> target` is a backedge -- in other words, if the target
    /// dominates the source.
    fn is_back_edge(&self, source: Location, target: Location) -> bool {
        target.dominates(source, &self.dominators)
    }

    /// Determine how the borrow was later used.
    /// First span returned points to the location of the conflicting use
    /// Second span if `Some` is returned in the case of closures and points
    /// to the use of the path
    fn later_use_kind(
        &self,
        borrow: &BorrowData<'tcx>,
        use_spans: UseSpans<'tcx>,
        location: Location,
    ) -> (LaterUseKind, Span, Option<Span>) {
        match use_spans {
            UseSpans::ClosureUse { capture_kind_span, path_span, .. } => {
                // Used in a closure.
                (LaterUseKind::ClosureCapture, capture_kind_span, Some(path_span))
            }
            UseSpans::PatUse(span)
            | UseSpans::OtherUse(span)
            | UseSpans::FnSelfUse { var_span: span, .. } => {
                let block = &self.body.basic_blocks[location.block];

                let kind = if let Some(&Statement {
                    kind: StatementKind::FakeRead(box (FakeReadCause::ForLet(_), _)),
                    ..
                }) = block.statements.get(location.statement_index)
                {
                    LaterUseKind::FakeLetRead
                } else if self.was_captured_by_trait_object(borrow) {
                    LaterUseKind::TraitCapture
                } else if location.statement_index == block.statements.len() {
                    if let TerminatorKind::Call { ref func, from_hir_call: true, .. } =
                        block.terminator().kind
                    {
                        // Just point to the function, to reduce the chance of overlapping spans.
                        let function_span = match func {
                            Operand::Constant(c) => c.span,
                            Operand::Copy(place) | Operand::Move(place) => {
                                if let Some(l) = place.as_local() {
                                    let local_decl = &self.body.local_decls[l];
                                    if self.local_names[l].is_none() {
                                        local_decl.source_info.span
                                    } else {
                                        span
                                    }
                                } else {
                                    span
                                }
                            }
                        };
                        return (LaterUseKind::Call, function_span, None);
                    } else {
                        LaterUseKind::Other
                    }
                } else {
                    LaterUseKind::Other
                };

                (kind, span, None)
            }
        }
    }

    /// Checks if a borrowed value was captured by a trait object. We do this by
    /// looking forward in the MIR from the reserve location and checking if we see
    /// an unsized cast to a trait object on our data.
    fn was_captured_by_trait_object(&self, borrow: &BorrowData<'tcx>) -> bool {
        // Start at the reserve location, find the place that we want to see cast to a trait object.
        let location = borrow.reserve_location;
        let block = &self.body[location.block];
        let stmt = block.statements.get(location.statement_index);
        debug!("was_captured_by_trait_object: location={:?} stmt={:?}", location, stmt);

        // We make a `queue` vector that has the locations we want to visit. As of writing, this
        // will only ever have one item at any given time, but by using a vector, we can pop from
        // it which simplifies the termination logic.
        let mut queue = vec![location];
        let mut target = if let Some(&Statement {
            kind: StatementKind::Assign(box (ref place, _)),
            ..
        }) = stmt
        {
            if let Some(local) = place.as_local() {
                local
            } else {
                return false;
            }
        } else {
            return false;
        };

        debug!("was_captured_by_trait: target={:?} queue={:?}", target, queue);
        while let Some(current_location) = queue.pop() {
            debug!("was_captured_by_trait: target={:?}", target);
            let block = &self.body[current_location.block];
            // We need to check the current location to find out if it is a terminator.
            let is_terminator = current_location.statement_index == block.statements.len();
            if !is_terminator {
                let stmt = &block.statements[current_location.statement_index];
                debug!("was_captured_by_trait_object: stmt={:?}", stmt);

                // The only kind of statement that we care about is assignments...
                if let StatementKind::Assign(box (place, rvalue)) = &stmt.kind {
                    let Some(into) = place.local_or_deref_local() else {
                        // Continue at the next location.
                        queue.push(current_location.successor_within_block());
                        continue;
                    };

                    match rvalue {
                        // If we see a use, we should check whether it is our data, and if so
                        // update the place that we're looking for to that new place.
                        Rvalue::Use(operand) => match operand {
                            Operand::Copy(place) | Operand::Move(place) => {
                                if let Some(from) = place.as_local() {
                                    if from == target {
                                        target = into;
                                    }
                                }
                            }
                            _ => {}
                        },
                        // If we see an unsized cast, then if it is our data we should check
                        // whether it is being cast to a trait object.
                        Rvalue::Cast(CastKind::Pointer(PointerCast::Unsize), operand, ty) => {
                            match operand {
                                Operand::Copy(place) | Operand::Move(place) => {
                                    if let Some(from) = place.as_local() {
                                        if from == target {
                                            debug!("was_captured_by_trait_object: ty={:?}", ty);
                                            // Check the type for a trait object.
                                            return match ty.kind() {
                                                // `&dyn Trait`
                                                ty::Ref(_, ty, _) if ty.is_trait() => true,
                                                // `Box<dyn Trait>`
                                                _ if ty.is_box() && ty.boxed_ty().is_trait() => {
                                                    true
                                                }
                                                // `dyn Trait`
                                                _ if ty.is_trait() => true,
                                                // Anything else.
                                                _ => false,
                                            };
                                        }
                                    }
                                    return false;
                                }
                                _ => return false,
                            }
                        }
                        _ => {}
                    }
                }

                // Continue at the next location.
                queue.push(current_location.successor_within_block());
            } else {
                // The only thing we need to do for terminators is progress to the next block.
                let terminator = block.terminator();
                debug!("was_captured_by_trait_object: terminator={:?}", terminator);

                if let TerminatorKind::Call { destination, target: Some(block), args, .. } =
                    &terminator.kind
                {
                    if let Some(dest) = destination.as_local() {
                        debug!(
                            "was_captured_by_trait_object: target={:?} dest={:?} args={:?}",
                            target, dest, args
                        );
                        // Check if one of the arguments to this function is the target place.
                        let found_target = args.iter().any(|arg| {
                            if let Operand::Move(place) = arg {
                                if let Some(potential) = place.as_local() {
                                    potential == target
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        });

                        // If it is, follow this to the next block and update the target.
                        if found_target {
                            target = dest;
                            queue.push(block.start_location());
                        }
                    }
                }
            }

            debug!("was_captured_by_trait: queue={:?}", queue);
        }

        // We didn't find anything and ran out of locations to check.
        false
    }
}
