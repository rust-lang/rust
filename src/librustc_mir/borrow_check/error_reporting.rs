// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::{WriteKind, StorageDeadOrDrop};
use borrow_check::prefixes::IsPrefixOf;
use borrow_check::nll::explain_borrow::BorrowExplanation;
use rustc::middle::region::ScopeTree;
use rustc::mir::{
    AggregateKind, BindingForm, BorrowKind, ClearCrossCrate, FakeReadCause, Field, Local,
    LocalDecl, LocalKind, Location, Operand, Place, ProjectionElem, Rvalue, Statement,
    StatementKind, VarBindingForm,
};
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::ty;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc::util::ppaux::with_highlight_region_for_bound_region;
use syntax_pos::Span;

use super::borrow_set::BorrowData;
use super::{Context, MirBorrowckCtxt};
use super::{InitializationRequiringAction, PrefixSet};

use dataflow::drop_flag_effects;
use dataflow::move_paths::indexes::MoveOutIndex;
use dataflow::move_paths::MovePathIndex;
use util::borrowck_errors::{BorrowckErrors, Origin};

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    pub(super) fn report_use_of_moved_or_uninitialized(
        &mut self,
        context: Context,
        desired_action: InitializationRequiringAction,
        (place, span): (&Place<'tcx>, Span),
        mpi: MovePathIndex,
    ) {
        debug!(
            "report_use_of_moved_or_uninitialized: context={:?} desired_action={:?} place={:?} \
            span={:?} mpi={:?}",
            context, desired_action, place, span, mpi
        );

        let use_spans = self
            .move_spans(place, context.loc)
            .or_else(|| self.borrow_spans(span, context.loc));
        let span = use_spans.args_or_use();

        let mois = self.get_moved_indexes(context, mpi);
        debug!("report_use_of_moved_or_uninitialized: mois={:?}", mois);

        if mois.is_empty() {
            let root_place = self.prefixes(&place, PrefixSet::All).last().unwrap();

            if self.uninitialized_error_reported.contains(&root_place.clone()) {
                debug!(
                    "report_use_of_moved_or_uninitialized place: error about {:?} suppressed",
                    root_place
                );
                return;
            }

            self.uninitialized_error_reported.insert(root_place.clone());

            let item_msg = match self.describe_place_with_options(place, IncludingDowncast(true)) {
                Some(name) => format!("`{}`", name),
                None => "value".to_owned(),
            };
            let mut err = self.infcx.tcx.cannot_act_on_uninitialized_variable(
                span,
                desired_action.as_noun(),
                &self
                    .describe_place_with_options(place, IncludingDowncast(true))
                    .unwrap_or("_".to_owned()),
                Origin::Mir,
            );
            err.span_label(span, format!("use of possibly uninitialized {}", item_msg));

            use_spans.var_span_label(
                &mut err,
                format!("{} occurs due to use in closure", desired_action.as_noun()),
            );

            err.buffer(&mut self.errors_buffer);
        } else {
            if let Some((reported_place, _)) = self.move_error_reported.get(&mois) {
                if self.prefixes(&reported_place, PrefixSet::All).any(|p| p == place) {
                    debug!("report_use_of_moved_or_uninitialized place: error suppressed \
                           mois={:?}", mois);
                    return;
                }
            }

            let msg = ""; //FIXME: add "partially " or "collaterally "

            let mut err = self.infcx.tcx.cannot_act_on_moved_value(
                span,
                desired_action.as_noun(),
                msg,
                self.describe_place_with_options(&place, IncludingDowncast(true)),
                Origin::Mir,
            );

            let mut is_loop_move = false;
            for moi in &mois {
                let move_out = self.move_data.moves[*moi];
                let moved_place = &self.move_data.move_paths[move_out.path].place;

                let move_spans = self.move_spans(moved_place, move_out.source);
                let move_span = move_spans.args_or_use();

                let move_msg = if move_spans.for_closure() {
                    " into closure"
                } else {
                    ""
                };

                if span == move_span {
                    err.span_label(
                        span,
                        format!("value moved{} here in previous iteration of loop", move_msg),
                    );
                    is_loop_move = true;
                } else {
                    err.span_label(move_span, format!("value moved{} here", move_msg));
                    move_spans.var_span_label(&mut err, "variable moved due to use in closure");
                };
            }

            use_spans.var_span_label(
                &mut err,
                format!("{} occurs due to use in closure", desired_action.as_noun()),
            );

            if !is_loop_move {
                err.span_label(
                    span,
                    format!(
                        "value {} here after move",
                        desired_action.as_verb_in_past_tense()
                    ),
                );
            }

            if let Some(ty) = self.retrieve_type_for_place(place) {
                let needs_note = match ty.sty {
                    ty::Closure(id, _) => {
                        let tables = self.infcx.tcx.typeck_tables_of(id);
                        let node_id = self.infcx.tcx.hir.as_local_node_id(id).unwrap();
                        let hir_id = self.infcx.tcx.hir.node_to_hir_id(node_id);
                        if tables.closure_kind_origins().get(hir_id).is_some() {
                            false
                        } else {
                            true
                        }
                    }
                    _ => true,
                };

                if needs_note {
                    let mpi = self.move_data.moves[mois[0]].path;
                    let place = &self.move_data.move_paths[mpi].place;

                    if let Some(ty) = self.retrieve_type_for_place(place) {
                        let note_msg = match self
                            .describe_place_with_options(place, IncludingDowncast(true))
                        {
                            Some(name) => format!("`{}`", name),
                            None => "value".to_owned(),
                        };

                        err.note(&format!(
                            "move occurs because {} has type `{}`, \
                             which does not implement the `Copy` trait",
                            note_msg, ty
                        ));
                    }
                }
            }

            if let Some((_, mut old_err)) = self.move_error_reported.insert(
                mois,
                (place.clone(), err)
            ) {
                // Cancel the old error so it doesn't ICE.
                old_err.cancel();
            }
        }
    }

    pub(super) fn report_move_out_while_borrowed(
        &mut self,
        context: Context,
        (place, _span): (&Place<'tcx>, Span),
        borrow: &BorrowData<'tcx>,
    ) {
        let tcx = self.infcx.tcx;
        let value_msg = match self.describe_place(place) {
            Some(name) => format!("`{}`", name),
            None => "value".to_owned(),
        };
        let borrow_msg = match self.describe_place(&borrow.borrowed_place) {
            Some(name) => format!("`{}`", name),
            None => "value".to_owned(),
        };

        let borrow_spans = self.retrieve_borrow_spans(borrow);
        let borrow_span = borrow_spans.args_or_use();

        let move_spans = self.move_spans(place, context.loc);
        let span = move_spans.args_or_use();

        let mut err = tcx.cannot_move_when_borrowed(
            span,
            &self.describe_place(place).unwrap_or("_".to_owned()),
            Origin::Mir,
        );
        err.span_label(borrow_span, format!("borrow of {} occurs here", borrow_msg));
        err.span_label(span, format!("move out of {} occurs here", value_msg));

        borrow_spans.var_span_label(&mut err, "borrow occurs due to use in closure");

        move_spans.var_span_label(&mut err, "move occurs due to use in closure");

        self.explain_why_borrow_contains_point(context, borrow, None)
            .emit(self.infcx.tcx, &mut err);
        err.buffer(&mut self.errors_buffer);
    }

    pub(super) fn report_use_while_mutably_borrowed(
        &mut self,
        context: Context,
        (place, _span): (&Place<'tcx>, Span),
        borrow: &BorrowData<'tcx>,
    ) {
        let tcx = self.infcx.tcx;

        let borrow_spans = self.retrieve_borrow_spans(borrow);
        let borrow_span = borrow_spans.args_or_use();

        // Conflicting borrows are reported separately, so only check for move
        // captures.
        let use_spans = self.move_spans(place, context.loc);
        let span = use_spans.var_or_use();

        let mut err = tcx.cannot_use_when_mutably_borrowed(
            span,
            &self.describe_place(place).unwrap_or("_".to_owned()),
            borrow_span,
            &self
                .describe_place(&borrow.borrowed_place)
                .unwrap_or("_".to_owned()),
            Origin::Mir,
        );

        borrow_spans.var_span_label(&mut err, {
            let place = &borrow.borrowed_place;
            let desc_place = self.describe_place(place).unwrap_or("_".to_owned());

            format!("borrow occurs due to use of `{}` in closure", desc_place)
        });

        self.explain_why_borrow_contains_point(context, borrow, None)
            .emit(self.infcx.tcx, &mut err);
        err.buffer(&mut self.errors_buffer);
    }

    pub(super) fn report_conflicting_borrow(
        &mut self,
        context: Context,
        (place, span): (&Place<'tcx>, Span),
        gen_borrow_kind: BorrowKind,
        issued_borrow: &BorrowData<'tcx>,
    ) {
        let issued_spans = self.retrieve_borrow_spans(issued_borrow);
        let issued_span = issued_spans.args_or_use();

        let borrow_spans = self.borrow_spans(span, context.loc);
        let span = borrow_spans.args_or_use();

        let desc_place = self.describe_place(place).unwrap_or("_".to_owned());
        let tcx = self.infcx.tcx;

        // FIXME: supply non-"" `opt_via` when appropriate
        let mut err = match (
            gen_borrow_kind,
            "immutable",
            "mutable",
            issued_borrow.kind,
            "immutable",
            "mutable",
        ) {
            (BorrowKind::Shared, lft, _, BorrowKind::Mut { .. }, _, rgt)
            | (BorrowKind::Mut { .. }, _, lft, BorrowKind::Shared, rgt, _) => tcx
                .cannot_reborrow_already_borrowed(
                    span,
                    &desc_place,
                    "",
                    lft,
                    issued_span,
                    "it",
                    rgt,
                    "",
                    None,
                    Origin::Mir,
                ),

            (BorrowKind::Mut { .. }, _, _, BorrowKind::Mut { .. }, _, _) => tcx
                .cannot_mutably_borrow_multiply(
                    span,
                    &desc_place,
                    "",
                    issued_span,
                    "",
                    None,
                    Origin::Mir,
                ),

            (BorrowKind::Unique, _, _, BorrowKind::Unique, _, _) => tcx
                .cannot_uniquely_borrow_by_two_closures(
                    span,
                    &desc_place,
                    issued_span,
                    None,
                    Origin::Mir,
                ),

            (BorrowKind::Unique, _, _, _, _, _) => tcx.cannot_uniquely_borrow_by_one_closure(
                span,
                &desc_place,
                "",
                issued_span,
                "it",
                "",
                None,
                Origin::Mir,
            ),

            (BorrowKind::Shared, lft, _, BorrowKind::Unique, _, _) => tcx
                .cannot_reborrow_already_uniquely_borrowed(
                    span,
                    &desc_place,
                    "",
                    lft,
                    issued_span,
                    "",
                    None,
                    Origin::Mir,
                ),

            (BorrowKind::Mut { .. }, _, lft, BorrowKind::Unique, _, _) => tcx
                .cannot_reborrow_already_uniquely_borrowed(
                    span,
                    &desc_place,
                    "",
                    lft,
                    issued_span,
                    "",
                    None,
                    Origin::Mir,
                ),

            (BorrowKind::Shared, _, _, BorrowKind::Shared, _, _) => unreachable!(),
        };

        if issued_spans == borrow_spans {
            borrow_spans.var_span_label(
                &mut err,
                format!("borrows occur due to use of `{}` in closure", desc_place),
            );
        } else {
            let borrow_place = &issued_borrow.borrowed_place;
            let borrow_place_desc = self.describe_place(borrow_place).unwrap_or("_".to_owned());
            issued_spans.var_span_label(
                &mut err,
                format!(
                    "first borrow occurs due to use of `{}` in closure",
                    borrow_place_desc
                ),
            );

            borrow_spans.var_span_label(
                &mut err,
                format!(
                    "second borrow occurs due to use of `{}` in closure",
                    desc_place
                ),
            );
        }

        self.explain_why_borrow_contains_point(context, issued_borrow, None)
            .emit(self.infcx.tcx, &mut err);

        err.buffer(&mut self.errors_buffer);
    }

    /// Reports StorageDeadOrDrop of `place` conflicts with `borrow`.
    ///
    /// This means that some data referenced by `borrow` needs to live
    /// past the point where the StorageDeadOrDrop of `place` occurs.
    /// This is usually interpreted as meaning that `place` has too
    /// short a lifetime. (But sometimes it is more useful to report
    /// it as a more direct conflict between the execution of a
    /// `Drop::drop` with an aliasing borrow.)
    pub(super) fn report_borrowed_value_does_not_live_long_enough(
        &mut self,
        context: Context,
        borrow: &BorrowData<'tcx>,
        place_span: (&Place<'tcx>, Span),
        kind: Option<WriteKind>,
    ) {
        debug!("report_borrowed_value_does_not_live_long_enough(\
                {:?}, {:?}, {:?}, {:?}\
                )",
               context, borrow, place_span, kind
        );

        let drop_span = place_span.1;
        let scope_tree = self.infcx.tcx.region_scope_tree(self.mir_def_id);
        let root_place = self
            .prefixes(&borrow.borrowed_place, PrefixSet::All)
            .last()
            .unwrap();

        let borrow_spans = self.retrieve_borrow_spans(borrow);
        let borrow_span = borrow_spans.var_or_use();

        let proper_span = match *root_place {
            Place::Local(local) => self.mir.local_decls[local].source_info.span,
            _ => drop_span,
        };

        if self
            .access_place_error_reported
            .contains(&(root_place.clone(), borrow_span))
        {
            debug!(
                "suppressing access_place error when borrow doesn't live long enough for {:?}",
                borrow_span
            );
            return;
        }

        self.access_place_error_reported
            .insert((root_place.clone(), borrow_span));

        if let Some(WriteKind::StorageDeadOrDrop(StorageDeadOrDrop::Destructor)) = kind {
            // If a borrow of path `B` conflicts with drop of `D` (and
            // we're not in the uninteresting case where `B` is a
            // prefix of `D`), then report this as a more interesting
            // destructor conflict.
            if !borrow.borrowed_place.is_prefix_of(place_span.0) {
                self.report_borrow_conflicts_with_destructor(context, borrow, place_span, kind);
                return;
            }
        }

        let err = match &self.describe_place(&borrow.borrowed_place) {
            Some(_) if self.is_place_thread_local(root_place) =>
                self.report_thread_local_value_does_not_live_long_enough(drop_span, borrow_span),
            Some(name) => self.report_local_value_does_not_live_long_enough(
                context,
                name,
                &scope_tree,
                &borrow,
                drop_span,
                borrow_spans,
                kind.map(|k| (k, place_span.0)),
            ),
            None => self.report_temporary_value_does_not_live_long_enough(
                context,
                &scope_tree,
                &borrow,
                drop_span,
                borrow_spans,
                proper_span,
            ),
        };

        err.buffer(&mut self.errors_buffer);
    }

    fn report_local_value_does_not_live_long_enough(
        &mut self,
        context: Context,
        name: &String,
        scope_tree: &Lrc<ScopeTree>,
        borrow: &BorrowData<'tcx>,
        drop_span: Span,
        borrow_spans: UseSpans,
        kind_place: Option<(WriteKind, &Place<'tcx>)>,
    ) -> DiagnosticBuilder<'cx> {
        debug!(
            "report_local_value_does_not_live_long_enough(\
             {:?}, {:?}, {:?}, {:?}, {:?}, {:?}\
             )",
            context, name, scope_tree, borrow, drop_span, borrow_spans
        );

        let borrow_span = borrow_spans.var_or_use();
        let mut err = self.infcx.tcx.path_does_not_live_long_enough(
            borrow_span,
            &format!("`{}`", name),
            Origin::Mir,
        );

        let explanation = self.explain_why_borrow_contains_point(context, borrow, kind_place);
        if let Some(annotation) = self.annotate_argument_and_return_for_borrow(borrow) {
            let region_name = annotation.emit(&mut err);

            err.span_label(
                borrow_span,
                format!("`{}` would have to be valid for `{}`", name, region_name)
            );

            if let Some(fn_node_id) = self.infcx.tcx.hir.as_local_node_id(self.mir_def_id) {
                err.span_label(
                    drop_span,
                    format!(
                        "but `{}` will be dropped here, when the function `{}` returns",
                        name, self.infcx.tcx.hir.name(fn_node_id),
                    )
                );

                err.note(
                    "functions cannot return a borrow to data owned within the function's scope, \
                     functions can only return borrows to data passed as arguments",
                );
                err.note(
                    "to learn more, visit <https://doc.rust-lang.org/book/second-edition/ch04-02-\
                     references-and-borrowing.html#dangling-references>",
                );
            } else {
                err.span_label(
                    drop_span,
                    format!("...but `{}` dropped here while still borrowed", name)
                );
            }

            if let BorrowExplanation::MustBeValidFor(..) = explanation { } else {
                explanation.emit(self.infcx.tcx, &mut err);
            }
        } else {
            err.span_label(borrow_span, "borrowed value does not live long enough");
            err.span_label(drop_span, format!("`{}` dropped here while still borrowed", name));

            borrow_spans.args_span_label(&mut err, "value captured here");

            explanation.emit(self.infcx.tcx, &mut err);
        }

        err
    }

    pub(super) fn report_borrow_conflicts_with_destructor(
        &mut self,
        context: Context,
        borrow: &BorrowData<'tcx>,
        (place, drop_span): (&Place<'tcx>, Span),
        kind: Option<WriteKind>,
    ) {
        debug!(
            "report_borrow_conflicts_with_destructor(\
             {:?}, {:?}, ({:?}, {:?}), {:?}\
             )",
            context, borrow, place, drop_span, kind,
        );

        let borrow_spans = self.retrieve_borrow_spans(borrow);
        let borrow_span = borrow_spans.var_or_use();

        let mut err = self.infcx.tcx.cannot_borrow_across_destructor(borrow_span, Origin::Mir);

        let (what_was_dropped, dropped_ty) = {
            let desc = match self.describe_place(place) {
                Some(name) => format!("`{}`", name.as_str()),
                None => format!("temporary value"),
            };
            let ty = place.ty(self.mir, self.infcx.tcx).to_ty(self.infcx.tcx);
            (desc, ty)
        };

        let label = match dropped_ty.sty {
            ty::Adt(adt, _) if adt.has_dtor(self.infcx.tcx) && !adt.is_box() => {
                match self.describe_place(&borrow.borrowed_place) {
                    Some(borrowed) =>
                        format!("here, drop of {D} needs exclusive access to `{B}`, \
                                 because the type `{T}` implements the `Drop` trait",
                                D=what_was_dropped, T=dropped_ty, B=borrowed),
                    None =>
                        format!("here is drop of {D}; whose type `{T}` implements the `Drop` trait",
                                D=what_was_dropped, T=dropped_ty),
                }
            }
            _ => format!("drop of {D} occurs here", D=what_was_dropped),
        };
        err.span_label(drop_span, label);

        // Only give this note and suggestion if they could be relevant.
        let explanation = self.explain_why_borrow_contains_point(
            context, borrow, kind.map(|k| (k, place)),
        );
        match explanation {
            BorrowExplanation::UsedLater {..} |
            BorrowExplanation::UsedLaterWhenDropped {..} => {
                err.note("consider using a `let` binding to create a longer lived value");
            },
            _ => {},
        }

        explanation.emit(self.infcx.tcx, &mut err);

        err.buffer(&mut self.errors_buffer);
    }

    fn report_thread_local_value_does_not_live_long_enough(
        &mut self,
        drop_span: Span,
        borrow_span: Span,
    ) -> DiagnosticBuilder<'cx> {
        debug!(
            "report_thread_local_value_does_not_live_long_enough(\
             {:?}, {:?}\
             )",
            drop_span, borrow_span
        );

        let mut err = self.infcx.tcx.thread_local_value_does_not_live_long_enough(
            borrow_span, Origin::Mir
        );

        err.span_label(
            borrow_span,
            "thread-local variables cannot be borrowed beyond the end of the function",
        );
        err.span_label(drop_span, "end of enclosing function is here");

        err
    }

    fn report_temporary_value_does_not_live_long_enough(
        &mut self,
        context: Context,
        scope_tree: &Lrc<ScopeTree>,
        borrow: &BorrowData<'tcx>,
        drop_span: Span,
        borrow_spans: UseSpans,
        proper_span: Span,
    ) -> DiagnosticBuilder<'cx> {
        debug!(
            "report_temporary_value_does_not_live_long_enough(\
             {:?}, {:?}, {:?}, {:?}, {:?}\
             )",
            context, scope_tree, borrow, drop_span, proper_span
        );

        let tcx = self.infcx.tcx;
        let mut err =
            tcx.path_does_not_live_long_enough(proper_span, "borrowed value", Origin::Mir);
        err.span_label(proper_span, "temporary value does not live long enough");
        err.span_label(drop_span, "temporary value only lives until here");

        let explanation = self.explain_why_borrow_contains_point(context, borrow, None);
        match explanation {
            BorrowExplanation::UsedLater(..) |
            BorrowExplanation::UsedLaterInLoop(..) |
            BorrowExplanation::UsedLaterWhenDropped(..) => {
                // Only give this note and suggestion if it could be relevant.
                err.note("consider using a `let` binding to create a longer lived value");
            },
            _ => {},
        }
        explanation.emit(self.infcx.tcx, &mut err);

        borrow_spans.args_span_label(&mut err, "value captured here");

        err
    }

    fn get_moved_indexes(&mut self, context: Context, mpi: MovePathIndex) -> Vec<MoveOutIndex> {
        let mir = self.mir;

        let mut stack = Vec::new();
        stack.extend(mir.predecessor_locations(context.loc));

        let mut visited = FxHashSet();
        let mut result = vec![];

        'dfs: while let Some(l) = stack.pop() {
            debug!(
                "report_use_of_moved_or_uninitialized: current_location={:?}",
                l
            );

            if !visited.insert(l) {
                continue;
            }

            // check for moves
            let stmt_kind = mir[l.block]
                .statements
                .get(l.statement_index)
                .map(|s| &s.kind);
            if let Some(StatementKind::StorageDead(..)) = stmt_kind {
                // this analysis only tries to find moves explicitly
                // written by the user, so we ignore the move-outs
                // created by `StorageDead` and at the beginning
                // of a function.
            } else {
                // If we are found a use of a.b.c which was in error, then we want to look for
                // moves not only of a.b.c but also a.b and a.
                //
                // Note that the moves data already includes "parent" paths, so we don't have to
                // worry about the other case: that is, if there is a move of a.b.c, it is already
                // marked as a move of a.b and a as well, so we will generate the correct errors
                // there.
                let mut mpis = vec![mpi];
                let move_paths = &self.move_data.move_paths;
                mpis.extend(move_paths[mpi].parents(move_paths));

                for moi in &self.move_data.loc_map[l] {
                    debug!("report_use_of_moved_or_uninitialized: moi={:?}", moi);
                    if mpis.contains(&self.move_data.moves[*moi].path) {
                        debug!("report_use_of_moved_or_uninitialized: found");
                        result.push(*moi);

                        // Strictly speaking, we could continue our DFS here. There may be
                        // other moves that can reach the point of error. But it is kind of
                        // confusing to highlight them.
                        //
                        // Example:
                        //
                        // ```
                        // let a = vec![];
                        // let b = a;
                        // let c = a;
                        // drop(a); // <-- current point of error
                        // ```
                        //
                        // Because we stop the DFS here, we only highlight `let c = a`,
                        // and not `let b = a`. We will of course also report an error at
                        // `let c = a` which highlights `let b = a` as the move.
                        continue 'dfs;
                    }
                }
            }

            // check for inits
            let mut any_match = false;
            drop_flag_effects::for_location_inits(self.infcx.tcx, self.mir, self.move_data, l, |m| {
                if m == mpi {
                    any_match = true;
                }
            });
            if any_match {
                continue 'dfs;
            }

            stack.extend(mir.predecessor_locations(l));
        }

        result
    }

    pub(super) fn report_illegal_mutation_of_borrowed(
        &mut self,
        context: Context,
        (place, span): (&Place<'tcx>, Span),
        loan: &BorrowData<'tcx>,
    ) {
        let loan_spans = self.retrieve_borrow_spans(loan);
        let loan_span = loan_spans.args_or_use();

        let tcx = self.infcx.tcx;
        let mut err = tcx.cannot_assign_to_borrowed(
            span,
            loan_span,
            &self.describe_place(place).unwrap_or("_".to_owned()),
            Origin::Mir,
        );

        loan_spans.var_span_label(&mut err, "borrow occurs due to use in closure");

        self.explain_why_borrow_contains_point(context, loan, None).emit(self.infcx.tcx, &mut err);

        err.buffer(&mut self.errors_buffer);
    }

    /// Reports an illegal reassignment; for example, an assignment to
    /// (part of) a non-`mut` local that occurs potentially after that
    /// local has already been initialized. `place` is the path being
    /// assigned; `err_place` is a place providing a reason why
    /// `place` is not mutable (e.g. the non-`mut` local `x` in an
    /// assignment to `x.f`).
    pub(super) fn report_illegal_reassignment(
        &mut self,
        _context: Context,
        (place, span): (&Place<'tcx>, Span),
        assigned_span: Span,
        err_place: &Place<'tcx>,
    ) {
        let (from_arg, local_decl) = if let Place::Local(local) = *err_place {
            if let LocalKind::Arg = self.mir.local_kind(local) {
                (true, Some(&self.mir.local_decls[local]))
            } else {
                (false, Some(&self.mir.local_decls[local]))
            }
        } else {
            (false, None)
        };

        // If root local is initialized immediately (everything apart from let
        // PATTERN;) then make the error refer to that local, rather than the
        // place being assigned later.
        let (place_description, assigned_span) = match local_decl {
            Some(LocalDecl {
                is_user_variable: Some(ClearCrossCrate::Clear),
                ..
            })
            | Some(LocalDecl {
                is_user_variable:
                    Some(ClearCrossCrate::Set(BindingForm::Var(VarBindingForm {
                        opt_match_place: None,
                        ..
                    }))),
                ..
            })
            | Some(LocalDecl {
                is_user_variable: None,
                ..
            })
            | None => (self.describe_place(place), assigned_span),
            Some(decl) => (self.describe_place(err_place), decl.source_info.span),
        };

        let mut err = self.infcx.tcx.cannot_reassign_immutable(
            span,
            place_description.as_ref().map(AsRef::as_ref).unwrap_or("_"),
            from_arg,
            Origin::Mir,
        );
        let msg = if from_arg {
            "cannot assign to immutable argument"
        } else {
            "cannot assign twice to immutable variable"
        };
        if span != assigned_span {
            if !from_arg {
                let value_msg = match place_description {
                    Some(name) => format!("`{}`", name),
                    None => "value".to_owned(),
                };
                err.span_label(assigned_span, format!("first assignment to {}", value_msg));
            }
        }
        if let Some(decl) = local_decl {
            if let Some(name) = decl.name {
                if decl.can_be_made_mutable() {
                    err.span_suggestion_with_applicability(
                        decl.source_info.span,
                        "make this binding mutable",
                        format!("mut {}", name),
                        Applicability::MachineApplicable,
                    );
                }
            }
        }
        err.span_label(span, msg);
        err.buffer(&mut self.errors_buffer);
    }
}

pub(super) struct IncludingDowncast(bool);

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    // End-user visible description of `place` if one can be found. If the
    // place is a temporary for instance, None will be returned.
    pub(super) fn describe_place(&self, place: &Place<'tcx>) -> Option<String> {
        self.describe_place_with_options(place, IncludingDowncast(false))
    }

    // End-user visible description of `place` if one can be found. If the
    // place is a temporary for instance, None will be returned.
    // `IncludingDowncast` parameter makes the function return `Err` if `ProjectionElem` is
    // `Downcast` and `IncludingDowncast` is true
    pub(super) fn describe_place_with_options(
        &self,
        place: &Place<'tcx>,
        including_downcast: IncludingDowncast,
    ) -> Option<String> {
        let mut buf = String::new();
        match self.append_place_to_string(place, &mut buf, false, &including_downcast) {
            Ok(()) => Some(buf),
            Err(()) => None,
        }
    }

    // Appends end-user visible description of `place` to `buf`.
    fn append_place_to_string(
        &self,
        place: &Place<'tcx>,
        buf: &mut String,
        mut autoderef: bool,
        including_downcast: &IncludingDowncast,
    ) -> Result<(), ()> {
        match *place {
            Place::Promoted(_) => {
                buf.push_str("promoted");
            }
            Place::Local(local) => {
                self.append_local_to_string(local, buf)?;
            }
            Place::Static(ref static_) => {
                buf.push_str(&self.infcx.tcx.item_name(static_.def_id).to_string());
            }
            Place::Projection(ref proj) => {
                match proj.elem {
                    ProjectionElem::Deref => {
                        let upvar_field_projection =
                            place.is_upvar_field_projection(self.mir, &self.infcx.tcx);
                        if let Some(field) = upvar_field_projection {
                            let var_index = field.index();
                            let name = self.mir.upvar_decls[var_index].debug_name.to_string();
                            if self.mir.upvar_decls[var_index].by_ref {
                                buf.push_str(&name);
                            } else {
                                buf.push_str(&format!("*{}", &name));
                            }
                        } else {
                            if autoderef {
                                self.append_place_to_string(
                                    &proj.base,
                                    buf,
                                    autoderef,
                                    &including_downcast,
                                )?;
                            } else if let Place::Local(local) = proj.base {
                                if let Some(ClearCrossCrate::Set(BindingForm::RefForGuard)) =
                                    self.mir.local_decls[local].is_user_variable
                                {
                                    self.append_place_to_string(
                                        &proj.base,
                                        buf,
                                        autoderef,
                                        &including_downcast,
                                    )?;
                                } else {
                                    buf.push_str(&"*");
                                    self.append_place_to_string(
                                        &proj.base,
                                        buf,
                                        autoderef,
                                        &including_downcast,
                                    )?;
                                }
                            } else {
                                buf.push_str(&"*");
                                self.append_place_to_string(
                                    &proj.base,
                                    buf,
                                    autoderef,
                                    &including_downcast,
                                )?;
                            }
                        }
                    }
                    ProjectionElem::Downcast(..) => {
                        self.append_place_to_string(
                            &proj.base,
                            buf,
                            autoderef,
                            &including_downcast,
                        )?;
                        if including_downcast.0 {
                            return Err(());
                        }
                    }
                    ProjectionElem::Field(field, _ty) => {
                        autoderef = true;

                        let upvar_field_projection =
                            place.is_upvar_field_projection(self.mir, &self.infcx.tcx);
                        if let Some(field) = upvar_field_projection {
                            let var_index = field.index();
                            let name = self.mir.upvar_decls[var_index].debug_name.to_string();
                            buf.push_str(&name);
                        } else {
                            let field_name = self.describe_field(&proj.base, field);
                            self.append_place_to_string(
                                &proj.base,
                                buf,
                                autoderef,
                                &including_downcast,
                            )?;
                            buf.push_str(&format!(".{}", field_name));
                        }
                    }
                    ProjectionElem::Index(index) => {
                        autoderef = true;

                        self.append_place_to_string(
                            &proj.base,
                            buf,
                            autoderef,
                            &including_downcast,
                        )?;
                        buf.push_str("[");
                        if self.append_local_to_string(index, buf).is_err() {
                            buf.push_str("..");
                        }
                        buf.push_str("]");
                    }
                    ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. } => {
                        autoderef = true;
                        // Since it isn't possible to borrow an element on a particular index and
                        // then use another while the borrow is held, don't output indices details
                        // to avoid confusing the end-user
                        self.append_place_to_string(
                            &proj.base,
                            buf,
                            autoderef,
                            &including_downcast,
                        )?;
                        buf.push_str(&"[..]");
                    }
                };
            }
        }

        Ok(())
    }

    // Appends end-user visible description of the `local` place to `buf`. If `local` doesn't have
    // a name, then `Err` is returned
    fn append_local_to_string(&self, local_index: Local, buf: &mut String) -> Result<(), ()> {
        let local = &self.mir.local_decls[local_index];
        match local.name {
            Some(name) => {
                buf.push_str(&name.to_string());
                Ok(())
            }
            None => Err(()),
        }
    }

    // End-user visible description of the `field`nth field of `base`
    fn describe_field(&self, base: &Place, field: Field) -> String {
        match *base {
            Place::Local(local) => {
                let local = &self.mir.local_decls[local];
                self.describe_field_from_ty(&local.ty, field)
            }
            Place::Promoted(ref prom) => self.describe_field_from_ty(&prom.1, field),
            Place::Static(ref static_) => self.describe_field_from_ty(&static_.ty, field),
            Place::Projection(ref proj) => match proj.elem {
                ProjectionElem::Deref => self.describe_field(&proj.base, field),
                ProjectionElem::Downcast(def, variant_index) => format!(
                    "{}",
                    def.variants[variant_index].fields[field.index()].ident
                ),
                ProjectionElem::Field(_, field_type) => {
                    self.describe_field_from_ty(&field_type, field)
                }
                ProjectionElem::Index(..)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. } => {
                    self.describe_field(&proj.base, field).to_string()
                }
            },
        }
    }

    // End-user visible description of the `field_index`nth field of `ty`
    fn describe_field_from_ty(&self, ty: &ty::Ty, field: Field) -> String {
        if ty.is_box() {
            // If the type is a box, the field is described from the boxed type
            self.describe_field_from_ty(&ty.boxed_ty(), field)
        } else {
            match ty.sty {
                ty::Adt(def, _) => if def.is_enum() {
                    field.index().to_string()
                } else {
                    def.non_enum_variant().fields[field.index()]
                        .ident
                        .to_string()
                },
                ty::Tuple(_) => field.index().to_string(),
                ty::Ref(_, ty, _) | ty::RawPtr(ty::TypeAndMut { ty, .. }) => {
                    self.describe_field_from_ty(&ty, field)
                }
                ty::Array(ty, _) | ty::Slice(ty) => self.describe_field_from_ty(&ty, field),
                ty::Closure(def_id, _) | ty::Generator(def_id, _, _) => {
                    // Convert the def-id into a node-id. node-ids are only valid for
                    // the local code in the current crate, so this returns an `Option` in case
                    // the closure comes from another crate. But in that case we wouldn't
                    // be borrowck'ing it, so we can just unwrap:
                    let node_id = self.infcx.tcx.hir.as_local_node_id(def_id).unwrap();
                    let freevar = self.infcx.tcx.with_freevars(node_id, |fv| fv[field.index()]);

                    self.infcx.tcx.hir.name(freevar.var_id()).to_string()
                }
                _ => {
                    // Might need a revision when the fields in trait RFC is implemented
                    // (https://github.com/rust-lang/rfcs/pull/1546)
                    bug!(
                        "End-user description not implemented for field access on `{:?}`",
                        ty.sty
                    );
                }
            }
        }
    }

    // Retrieve type of a place for the current MIR representation
    fn retrieve_type_for_place(&self, place: &Place<'tcx>) -> Option<ty::Ty> {
        match place {
            Place::Local(local) => {
                let local = &self.mir.local_decls[*local];
                Some(local.ty)
            }
            Place::Promoted(ref prom) => Some(prom.1),
            Place::Static(ref st) => Some(st.ty),
            Place::Projection(ref proj) => match proj.elem {
                ProjectionElem::Field(_, ty) => Some(ty),
                _ => None,
            },
        }
    }

    /// Check if a place is a thread-local static.
    pub fn is_place_thread_local(&self, place: &Place<'tcx>) -> bool {
        if let Place::Static(statik) = place {
            let attrs = self.infcx.tcx.get_attrs(statik.def_id);
            let is_thread_local = attrs.iter().any(|attr| attr.check_name("thread_local"));

            debug!(
                "is_place_thread_local: attrs={:?} is_thread_local={:?}",
                attrs, is_thread_local
            );
            is_thread_local
        } else {
            debug!("is_place_thread_local: no");
            false
        }
    }

    /// Returns the `FakeReadCause` at this location if it is a `FakeRead` statement.
    pub(super) fn retrieve_fake_read_cause_for_location(
        &self,
        location: &Location,
    ) -> Option<FakeReadCause> {
        let stmt = self.mir.basic_blocks()[location.block]
            .statements
            .get(location.statement_index)?;
        if let StatementKind::FakeRead(cause, _) = stmt.kind {
            Some(cause)
        } else {
            None
        }
    }

    /// Annotate argument and return type of function and closure with (synthesized) lifetime for
    /// borrow of local value that does not live long enough.
    fn annotate_argument_and_return_for_borrow(
        &self,
        borrow: &BorrowData<'tcx>,
    ) -> Option<AnnotatedBorrowFnSignature> {
        // There are two cases that need handled: when a closure is involved and
        // when a closure is not involved.
        let location = borrow.reserve_location;
        let is_closure = self.infcx.tcx.is_closure(self.mir_def_id);

        match self.mir[location.block].statements.get(location.statement_index) {
            // When a closure is involved, we expect the reserve location to be an assignment
            // to a temporary local, which will be followed by a closure.
            Some(&Statement {
                kind: StatementKind::Assign(Place::Local(local), _),
                ..
            }) if self.mir.local_kind(local) == LocalKind::Temp => {
                // Look for the statements within this block after assigning to a local to see
                // if we have a closure. If we do, then annotate it.
                for stmt in &self.mir[location.block].statements[location.statement_index + 1..] {
                    if let StatementKind::Assign(
                        _,
                        Rvalue::Aggregate(
                            box AggregateKind::Closure(def_id, substs),
                            _
                        )
                    ) = stmt.kind {
                        return self.annotate_fn_sig(
                            def_id,
                            self.infcx.closure_sig(def_id, substs)
                        );
                    }
                }
            }
            _ => {}
        }

        // If this is not the case, then return if we're currently on a closure (as we
        // don't have a substs to get the PolyFnSig) or attempt to get the arguments
        // and return type of the function.
        if is_closure {
            None
        } else {
            let ty = self.infcx.tcx.type_of(self.mir_def_id);
            match ty.sty {
                ty::TyKind::FnDef(_, _) | ty::TyKind::FnPtr(_) =>
                    self.annotate_fn_sig(
                        self.mir_def_id,
                        self.infcx.tcx.fn_sig(self.mir_def_id)
                    ),
                _ => None,
            }
        }
    }

    /// Annotate the first argument and return type of a function signature if they are
    /// references.
    fn annotate_fn_sig(
        &self,
        did: DefId,
        sig: ty::PolyFnSig<'tcx>,
    ) -> Option<AnnotatedBorrowFnSignature> {
        let is_closure = self.infcx.tcx.is_closure(did);
        let fn_node_id = self.infcx.tcx.hir.as_local_node_id(did)?;
        let fn_decl = self.infcx.tcx.hir.fn_decl(fn_node_id)?;

        // We need to work out which arguments to highlight. We do this by looking
        // at the return type, where there are three cases:
        //
        // 1. If there are named arguments, then we should highlight the return type and
        //    highlight any of the arguments that are also references with that lifetime.
        //    If there are no arguments that have the same lifetime as the return type,
        //    then don't highlight anything.
        // 2. The return type is a reference with an anonymous lifetime. If this is
        //    the case, then we can take advantage of (and teach) the lifetime elision
        //    rules.
        //
        //    We know that an error is being reported. So the arguments and return type
        //    must satisfy the elision rules. Therefore, if there is a single argument
        //    then that means the return type and first (and only) argument have the same
        //    lifetime and the borrow isn't meeting that, we can highlight the argument
        //    and return type.
        //
        //    If there are multiple arguments then the first argument must be self (else
        //    it would not satisfy the elision rules), so we can highlight self and the
        //    return type.
        // 3. The return type is not a reference. In this case, we don't highlight
        //    anything.
        let return_ty = sig.output();
        match return_ty.skip_binder().sty {
            ty::TyKind::Ref(return_region, _, _) if return_region.has_name() && !is_closure => {
                // This is case 1 from above, return type is a named reference so we need to
                // search for relevant arguments.
                let mut arguments = Vec::new();
                for (index, argument) in sig.inputs().skip_binder().iter().enumerate() {
                    if let ty::TyKind::Ref(argument_region, _, _) = argument.sty {
                        if argument_region == return_region {
                            // Need to use the `rustc::ty` types to compare against the
                            // `return_region`. Then use the `rustc::hir` type to get only
                            // the lifetime span.
                            match &fn_decl.inputs[index].node {
                                hir::TyKind::Rptr(lifetime, _) => {
                                    // With access to the lifetime, we can get
                                    // the span of it.
                                    arguments.push((*argument, lifetime.span));
                                },
                                _ => bug!("ty type is a ref but hir type is not"),
                            }
                        }
                    }
                }

                // We need to have arguments. This shouldn't happen, but it's worth checking.
                if arguments.is_empty() {
                    return None;
                }

                // We use a mix of the HIR and the Ty types to get information
                // as the HIR doesn't have full types for closure arguments.
                let return_ty = *sig.output().skip_binder();
                let mut return_span = fn_decl.output.span();
                if let hir::FunctionRetTy::Return(ty) = fn_decl.output {
                    if let hir::TyKind::Rptr(lifetime, _) = ty.into_inner().node {
                        return_span = lifetime.span;
                    }
                }

                Some(AnnotatedBorrowFnSignature::NamedFunction {
                    arguments,
                    return_ty,
                    return_span,
                })
            },
            ty::TyKind::Ref(_, _, _) if is_closure => {
                // This is case 2 from above but only for closures, return type is anonymous
                // reference so we select
                // the first argument.
                let argument_span = fn_decl.inputs.first()?.span;
                let argument_ty = sig.inputs().skip_binder().first()?;

                // Closure arguments are wrapped in a tuple, so we need to get the first
                // from that.
                if let ty::TyKind::Tuple(elems) = argument_ty.sty {
                    let argument_ty = elems.first()?;
                    if let ty::TyKind::Ref(_, _, _) = argument_ty.sty {
                        return Some(AnnotatedBorrowFnSignature::Closure {
                            argument_ty,
                            argument_span,
                        });
                    }
                }

                None
            },
            ty::TyKind::Ref(_, _, _) => {
                // This is also case 2 from above but for functions, return type is still an
                // anonymous reference so we select the first argument.
                let argument_span = fn_decl.inputs.first()?.span;
                let argument_ty = sig.inputs().skip_binder().first()?;

                let return_span = fn_decl.output.span();
                let return_ty = *sig.output().skip_binder();

                // We expect the first argument to be a reference.
                match argument_ty.sty {
                    ty::TyKind::Ref(_, _, _) => {},
                    _ => return None,
                }

                Some(AnnotatedBorrowFnSignature::AnonymousFunction {
                    argument_ty,
                    argument_span,
                    return_ty,
                    return_span,
                })
            },
            _ => {
                // This is case 3 from above, return type is not a reference so don't highlight
                // anything.
                None
            },
        }
    }
}

#[derive(Debug)]
enum AnnotatedBorrowFnSignature<'tcx> {
    NamedFunction {
        arguments: Vec<(ty::Ty<'tcx>, Span)>,
        return_ty: ty::Ty<'tcx>,
        return_span: Span,
    },
    AnonymousFunction {
        argument_ty: ty::Ty<'tcx>,
        argument_span: Span,
        return_ty: ty::Ty<'tcx>,
        return_span: Span,
    },
    Closure {
        argument_ty: ty::Ty<'tcx>,
        argument_span: Span,
    }
}

impl<'tcx> AnnotatedBorrowFnSignature<'tcx> {
    /// Annotate the provided diagnostic with information about borrow from the fn signature that
    /// helps explain.
    fn emit(
        &self,
        diag: &mut DiagnosticBuilder<'_>
    ) -> String {
        match self {
            AnnotatedBorrowFnSignature::Closure { argument_ty, argument_span } => {
                diag.span_label(
                    *argument_span,
                    format!("has type `{}`", self.get_name_for_ty(argument_ty, 0)),
                );

                self.get_region_name_for_ty(argument_ty, 0)
            },
            AnnotatedBorrowFnSignature::AnonymousFunction {
                argument_ty,
                argument_span,
                return_ty,
                return_span,
            } => {
                let argument_ty_name = self.get_name_for_ty(argument_ty, 0);
                diag.span_label(
                    *argument_span,
                    format!("has type `{}`", argument_ty_name)
                );

                let return_ty_name = self.get_name_for_ty(return_ty, 0);
                let types_equal = return_ty_name == argument_ty_name;
                diag.span_label(
                    *return_span,
                    format!(
                        "{}has type `{}`",
                        if types_equal { "also " } else { "" },
                        return_ty_name,
                    )
                );

                diag.note(
                    "argument and return type have the same lifetime due to lifetime elision rules",
                );
                diag.note(
                    "to learn more, visit <https://doc.rust-lang.org/book/second-edition/ch10-03-\
                     lifetime-syntax.html#lifetime-elision>",
                );

                self.get_region_name_for_ty(return_ty, 0)
            },
            AnnotatedBorrowFnSignature::NamedFunction {
                arguments,
                return_ty,
                return_span,
            } => {
                // Region of return type and arguments checked to be the same earlier.
                let region_name = self.get_region_name_for_ty(return_ty, 0);
                for (_, argument_span) in arguments {
                    diag.span_label(
                        *argument_span,
                        format!("has lifetime `{}`", region_name)
                    );
                }

                diag.span_label(
                    *return_span,
                    format!(
                        "also has lifetime `{}`",
                        region_name,
                    )
                );

                diag.help(
                    &format!(
                        "use data from the highlighted arguments which match the `{}` lifetime of \
                         the return type",
                         region_name,
                    ),
                );

                region_name
            },
        }
    }

    /// Return the name of the provided `Ty` (that must be a reference) with a synthesized lifetime
    /// name where required.
    fn get_name_for_ty(&self, ty: ty::Ty<'tcx>, counter: usize) -> String {
        // We need to add synthesized lifetimes where appropriate. We do
        // this by hooking into the pretty printer and telling it to label the
        // lifetimes without names with the value `'0`.
        match ty.sty {
            ty::TyKind::Ref(ty::RegionKind::ReLateBound(_, br), _, _) |
            ty::TyKind::Ref(ty::RegionKind::ReSkolemized(_, br), _, _) =>
                with_highlight_region_for_bound_region(*br, counter, || format!("{}", ty)),
            _ => format!("{}", ty),
        }
    }

    /// Return the name of the provided `Ty` (that must be a reference)'s region with a
    /// synthesized lifetime name where required.
    fn get_region_name_for_ty(&self, ty: ty::Ty<'tcx>, counter: usize) -> String {
        match ty.sty {
            ty::TyKind::Ref(region, _, _) => match region {
                ty::RegionKind::ReLateBound(_, br) |
                ty::RegionKind::ReSkolemized(_, br) =>
                    with_highlight_region_for_bound_region(*br, counter, || format!("{}", region)),
                _ => format!("{}", region),
            }
            _ => bug!("ty for annotation of borrow region is not a reference"),
        }
    }
}

// The span(s) associated to a use of a place.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum UseSpans {
    // The access is caused by capturing a variable for a closure.
    ClosureUse {
        // The span of the args of the closure, including the `move` keyword if
        // it's present.
        args_span: Span,
        // The span of the first use of the captured variable inside the closure.
        var_span: Span,
    },
    // This access has a single span associated to it: common case.
    OtherUse(Span),
}

impl UseSpans {
    pub(super) fn args_or_use(self) -> Span {
        match self {
            UseSpans::ClosureUse {
                args_span: span, ..
            }
            | UseSpans::OtherUse(span) => span,
        }
    }

    pub(super) fn var_or_use(self) -> Span {
        match self {
            UseSpans::ClosureUse { var_span: span, .. } | UseSpans::OtherUse(span) => span,
        }
    }

    // Add a span label to the arguments of the closure, if it exists.
    pub(super) fn args_span_label(self, err: &mut DiagnosticBuilder, message: impl Into<String>) {
        if let UseSpans::ClosureUse { args_span, .. } = self {
            err.span_label(args_span, message);
        }
    }

    // Add a span label to the use of the captured variable, if it exists.
    pub(super) fn var_span_label(self, err: &mut DiagnosticBuilder, message: impl Into<String>) {
        if let UseSpans::ClosureUse { var_span, .. } = self {
            err.span_label(var_span, message);
        }
    }

    pub(super) fn for_closure(self) -> bool {
        match self {
            UseSpans::ClosureUse { .. } => true,
            UseSpans::OtherUse(_) => false,
        }
    }

    pub(super) fn or_else<F>(self, if_other: F) -> Self
    where
        F: FnOnce() -> Self,
    {
        match self {
            closure @ UseSpans::ClosureUse { .. } => closure,
            UseSpans::OtherUse(_) => if_other(),
        }
    }
}

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    /// Finds the spans associated to a move or copy of move_place at location.
    pub(super) fn move_spans(
        &self,
        moved_place: &Place<'tcx>, // Could also be an upvar.
        location: Location,
    ) -> UseSpans {
        use self::UseSpans::*;
        use rustc::hir::ExprKind::Closure;
        use rustc::mir::AggregateKind;

        let stmt = match self.mir[location.block]
            .statements
            .get(location.statement_index)
        {
            Some(stmt) => stmt,
            None => return OtherUse(self.mir.source_info(location).span),
        };

        if let StatementKind::Assign(_, Rvalue::Aggregate(ref kind, ref places)) = stmt.kind {
            if let AggregateKind::Closure(def_id, _) = **kind {
                debug!("find_closure_move_span: found closure {:?}", places);

                if let Some(node_id) = self.infcx.tcx.hir.as_local_node_id(def_id) {
                    if let Closure(
                        _, _, _, args_span, _
                    ) = self.infcx.tcx.hir.expect_expr(node_id).node {
                        if let Some(var_span) = self.infcx.tcx.with_freevars(node_id, |freevars| {
                            for (v, place) in freevars.iter().zip(places) {
                                match place {
                                    Operand::Copy(place) | Operand::Move(place)
                                        if moved_place == place =>
                                    {
                                        debug!(
                                            "find_closure_move_span: found captured local {:?}",
                                            place
                                        );
                                        return Some(v.span);
                                    }
                                    _ => {}
                                }
                            }
                            None
                        }) {
                            return ClosureUse {
                                args_span,
                                var_span,
                            };
                        }
                    }
                }
            }
        }

        return OtherUse(stmt.source_info.span);
    }

    /// Finds the span of arguments of a closure (within `maybe_closure_span`)
    /// and its usage of the local assigned at `location`.
    /// This is done by searching in statements succeeding `location`
    /// and originating from `maybe_closure_span`.
    pub(super) fn borrow_spans(&self, use_span: Span, location: Location) -> UseSpans {
        use self::UseSpans::*;
        use rustc::hir::ExprKind::Closure;

        let local = match self.mir[location.block]
            .statements
            .get(location.statement_index)
        {
            Some(&Statement {
                kind: StatementKind::Assign(Place::Local(local), _),
                ..
            }) => local,
            _ => return OtherUse(use_span),
        };

        if self.mir.local_kind(local) != LocalKind::Temp {
            // operands are always temporaries.
            return OtherUse(use_span);
        }

        for stmt in &self.mir[location.block].statements[location.statement_index + 1..] {
            if let StatementKind::Assign(_, Rvalue::Aggregate(ref kind, ref places)) = stmt.kind {
                if let AggregateKind::Closure(def_id, _) = **kind {
                    debug!("find_closure_borrow_span: found closure {:?}", places);

                    return if let Some(node_id) = self.infcx.tcx.hir.as_local_node_id(def_id) {
                        let args_span = if let Closure(_, _, _, span, _) =
                            self.infcx.tcx.hir.expect_expr(node_id).node
                        {
                            span
                        } else {
                            return OtherUse(use_span);
                        };

                        self.infcx.tcx
                            .with_freevars(node_id, |freevars| {
                                for (v, place) in freevars.iter().zip(places) {
                                    match *place {
                                        Operand::Copy(Place::Local(l))
                                        | Operand::Move(Place::Local(l))
                                            if local == l =>
                                        {
                                            debug!(
                                                "find_closure_borrow_span: found captured local \
                                                 {:?}",
                                                l
                                            );
                                            return Some(v.span);
                                        }
                                        _ => {}
                                    }
                                }
                                None
                            }).map(|var_span| ClosureUse {
                                args_span,
                                var_span,
                            }).unwrap_or(OtherUse(use_span))
                    } else {
                        OtherUse(use_span)
                    };
                }
            }

            if use_span != stmt.source_info.span {
                break;
            }
        }

        OtherUse(use_span)
    }

    /// Helper to retrieve span(s) of given borrow from the current MIR
    /// representation
    pub(super) fn retrieve_borrow_spans(&self, borrow: &BorrowData) -> UseSpans {
        let span = self.mir.source_info(borrow.reserve_location).span;
        self.borrow_spans(span, borrow.reserve_location)
    }
}
