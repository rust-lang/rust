use crate::borrow_check::nll::explain_borrow::BorrowExplanation;
use crate::borrow_check::nll::region_infer::{RegionName, RegionNameSource};
use crate::borrow_check::prefixes::IsPrefixOf;
use crate::borrow_check::WriteKind;
use rustc::hir;
use rustc::hir::def::Namespace;
use rustc::hir::def_id::DefId;
use rustc::middle::region::ScopeTree;
use rustc::mir::{
    self, AggregateKind, BindingForm, BorrowKind, ClearCrossCrate, Constant,
    ConstraintCategory, Field, Local, LocalDecl, LocalKind, Location, Operand,
    Place, PlaceBase, PlaceProjection, ProjectionElem, Rvalue, Statement, StatementKind,
    Static, StaticKind, TerminatorKind, VarBindingForm,
};
use rustc::ty::{self, DefIdTree};
use rustc::ty::layout::VariantIdx;
use rustc::ty::print::Print;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{Applicability, DiagnosticBuilder};
use syntax_pos::Span;
use syntax::source_map::CompilerDesugaringKind;

use super::borrow_set::BorrowData;
use super::{Context, MirBorrowckCtxt};
use super::{InitializationRequiringAction, PrefixSet};
use crate::dataflow::drop_flag_effects;
use crate::dataflow::move_paths::indexes::MoveOutIndex;
use crate::dataflow::move_paths::MovePathIndex;
use crate::util::borrowck_errors::{BorrowckErrors, Origin};

#[derive(Debug)]
struct MoveSite {
    /// Index of the "move out" that we found. The `MoveData` can
    /// then tell us where the move occurred.
    moi: MoveOutIndex,

    /// `true` if we traversed a back edge while walking from the point
    /// of error to the move site.
    traversed_back_edge: bool
}

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    pub(super) fn report_use_of_moved_or_uninitialized(
        &mut self,
        context: Context,
        desired_action: InitializationRequiringAction,
        (moved_place, used_place, span): (&Place<'tcx>, &Place<'tcx>, Span),
        mpi: MovePathIndex,
    ) {
        debug!(
            "report_use_of_moved_or_uninitialized: context={:?} desired_action={:?} \
             moved_place={:?} used_place={:?} span={:?} mpi={:?}",
            context, desired_action, moved_place, used_place, span, mpi
        );

        let use_spans = self.move_spans(moved_place, context.loc)
            .or_else(|| self.borrow_spans(span, context.loc));
        let span = use_spans.args_or_use();

        let move_site_vec = self.get_moved_indexes(context, mpi);
        debug!(
            "report_use_of_moved_or_uninitialized: move_site_vec={:?}",
            move_site_vec
        );
        let move_out_indices: Vec<_> = move_site_vec
            .iter()
            .map(|move_site| move_site.moi)
            .collect();

        if move_out_indices.is_empty() {
            let root_place = self.prefixes(&used_place, PrefixSet::All).last().unwrap();

            if self.uninitialized_error_reported.contains(root_place) {
                debug!(
                    "report_use_of_moved_or_uninitialized place: error about {:?} suppressed",
                    root_place
                );
                return;
            }

            self.uninitialized_error_reported.insert(root_place.clone());

            let item_msg = match self.describe_place_with_options(used_place,
                                                                  IncludingDowncast(true)) {
                Some(name) => format!("`{}`", name),
                None => "value".to_owned(),
            };
            let mut err = self.infcx.tcx.cannot_act_on_uninitialized_variable(
                span,
                desired_action.as_noun(),
                &self.describe_place_with_options(moved_place, IncludingDowncast(true))
                    .unwrap_or_else(|| "_".to_owned()),
                Origin::Mir,
            );
            err.span_label(span, format!("use of possibly uninitialized {}", item_msg));

            use_spans.var_span_label(
                &mut err,
                format!("{} occurs due to use{}", desired_action.as_noun(), use_spans.describe()),
            );

            err.buffer(&mut self.errors_buffer);
        } else {
            if let Some((reported_place, _)) = self.move_error_reported.get(&move_out_indices) {
                if self.prefixes(&reported_place, PrefixSet::All)
                    .any(|p| p == used_place)
                {
                    debug!(
                        "report_use_of_moved_or_uninitialized place: error suppressed \
                         mois={:?}",
                        move_out_indices
                    );
                    return;
                }
            }

            let msg = ""; //FIXME: add "partially " or "collaterally "

            let mut err = self.infcx.tcx.cannot_act_on_moved_value(
                span,
                desired_action.as_noun(),
                msg,
                self.describe_place_with_options(&moved_place, IncludingDowncast(true)),
                Origin::Mir,
            );

            self.add_moved_or_invoked_closure_note(
                context.loc,
                used_place,
                &mut err,
            );

            let mut is_loop_move = false;
            let is_partial_move = move_site_vec.iter().any(|move_site| {
                let move_out = self.move_data.moves[(*move_site).moi];
                let moved_place = &self.move_data.move_paths[move_out.path].place;
                used_place != moved_place && used_place.is_prefix_of(moved_place)
            });
            for move_site in &move_site_vec {
                let move_out = self.move_data.moves[(*move_site).moi];
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
                        format!("value moved{} here, in previous iteration of loop", move_msg),
                    );
                    if Some(CompilerDesugaringKind::ForLoop) == span.compiler_desugaring_kind() {
                        if let Ok(snippet) = self.infcx.tcx.sess.source_map()
                            .span_to_snippet(span)
                        {
                            err.span_suggestion(
                                move_span,
                                "consider borrowing this to avoid moving it into the for loop",
                                format!("&{}", snippet),
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                    is_loop_move = true;
                } else if move_site.traversed_back_edge {
                    err.span_label(
                        move_span,
                        format!(
                            "value moved{} here, in previous iteration of loop",
                            move_msg
                        ),
                    );
                } else {
                    err.span_label(move_span, format!("value moved{} here", move_msg));
                    move_spans.var_span_label(
                        &mut err,
                        format!("variable moved due to use{}", move_spans.describe()),
                    );
                };
            }

            use_spans.var_span_label(
                &mut err,
                format!("{} occurs due to use{}", desired_action.as_noun(), use_spans.describe()),
            );

            if !is_loop_move {
                err.span_label(
                    span,
                    format!(
                        "value {} here {}",
                        desired_action.as_verb_in_past_tense(),
                        if is_partial_move { "after partial move" } else { "after move" },
                    ),
                );
            }

            let ty = used_place.ty(self.mir, self.infcx.tcx).ty;
            let needs_note = match ty.sty {
                ty::Closure(id, _) => {
                    let tables = self.infcx.tcx.typeck_tables_of(id);
                    let hir_id = self.infcx.tcx.hir().as_local_hir_id(id).unwrap();

                    tables.closure_kind_origins().get(hir_id).is_none()
                }
                _ => true,
            };

            if needs_note {
                let mpi = self.move_data.moves[move_out_indices[0]].path;
                let place = &self.move_data.move_paths[mpi].place;

                let ty = place.ty(self.mir, self.infcx.tcx).ty;
                let opt_name = self.describe_place_with_options(place, IncludingDowncast(true));
                let note_msg = match opt_name {
                    Some(ref name) => format!("`{}`", name),
                    None => "value".to_owned(),
                };
                if let ty::TyKind::Param(param_ty) = ty.sty {
                    let tcx = self.infcx.tcx;
                    let generics = tcx.generics_of(self.mir_def_id);
                    let def_id = generics.type_param(&param_ty, tcx).def_id;
                    if let Some(sp) = tcx.hir().span_if_local(def_id) {
                        err.span_label(
                            sp,
                            "consider adding a `Copy` constraint to this type argument",
                        );
                    }
                }
                if let Place::Base(PlaceBase::Local(local)) = place {
                    let decl = &self.mir.local_decls[*local];
                    err.span_label(
                        decl.source_info.span,
                        format!(
                            "move occurs because {} has type `{}`, \
                                which does not implement the `Copy` trait",
                            note_msg, ty,
                    ));
                } else {
                    err.note(&format!(
                        "move occurs because {} has type `{}`, \
                         which does not implement the `Copy` trait",
                        note_msg, ty
                    ));
                }
            }

            if let Some((_, mut old_err)) = self.move_error_reported
                .insert(move_out_indices, (used_place.clone(), err))
            {
                // Cancel the old error so it doesn't ICE.
                old_err.cancel();
            }
        }
    }

    pub(super) fn report_move_out_while_borrowed(
        &mut self,
        context: Context,
        (place, span): (&Place<'tcx>, Span),
        borrow: &BorrowData<'tcx>,
    ) {
        debug!(
            "report_move_out_while_borrowed: context={:?} place={:?} span={:?} borrow={:?}",
            context, place, span, borrow
        );
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
            &self.describe_place(place).unwrap_or_else(|| "_".to_owned()),
            Origin::Mir,
        );
        err.span_label(borrow_span, format!("borrow of {} occurs here", borrow_msg));
        err.span_label(span, format!("move out of {} occurs here", value_msg));

        borrow_spans.var_span_label(
            &mut err,
            format!("borrow occurs due to use{}", borrow_spans.describe())
        );

        move_spans.var_span_label(
            &mut err,
            format!("move occurs due to use{}", move_spans.describe())
        );

        self.explain_why_borrow_contains_point(
            context,
            borrow,
            None,
        ).add_explanation_to_diagnostic(self.infcx.tcx, self.mir, &mut err, "", Some(borrow_span));
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
            &self.describe_place(place).unwrap_or_else(|| "_".to_owned()),
            borrow_span,
            &self.describe_place(&borrow.borrowed_place)
                .unwrap_or_else(|| "_".to_owned()),
            Origin::Mir,
        );

        borrow_spans.var_span_label(&mut err, {
            let place = &borrow.borrowed_place;
            let desc_place = self.describe_place(place).unwrap_or_else(|| "_".to_owned());

            format!("borrow occurs due to use of `{}`{}", desc_place, borrow_spans.describe())
        });

        self.explain_why_borrow_contains_point(context, borrow, None)
            .add_explanation_to_diagnostic(self.infcx.tcx, self.mir, &mut err, "", None);
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

        let container_name = if issued_spans.for_generator() || borrow_spans.for_generator() {
            "generator"
        } else {
            "closure"
        };

        let (desc_place, msg_place, msg_borrow, union_type_name) =
            self.describe_place_for_conflicting_borrow(place, &issued_borrow.borrowed_place);

        let explanation = self.explain_why_borrow_contains_point(context, issued_borrow, None);
        let second_borrow_desc = if explanation.is_explained() {
            "second "
        } else {
            ""
        };

        // FIXME: supply non-"" `opt_via` when appropriate
        let tcx = self.infcx.tcx;
        let first_borrow_desc;
        let mut err = match (
            gen_borrow_kind,
            "immutable",
            "mutable",
            issued_borrow.kind,
            "immutable",
            "mutable",
        ) {
            (BorrowKind::Shared, lft, _, BorrowKind::Mut { .. }, _, rgt) => {
                first_borrow_desc = "mutable ";
                tcx.cannot_reborrow_already_borrowed(
                    span,
                    &desc_place,
                    &msg_place,
                    lft,
                    issued_span,
                    "it",
                    rgt,
                    &msg_borrow,
                    None,
                    Origin::Mir,
                )
            }
            (BorrowKind::Mut { .. }, _, lft, BorrowKind::Shared, rgt, _) => {
                first_borrow_desc = "immutable ";
                tcx.cannot_reborrow_already_borrowed(
                    span,
                    &desc_place,
                    &msg_place,
                    lft,
                    issued_span,
                    "it",
                    rgt,
                    &msg_borrow,
                    None,
                    Origin::Mir,
                )
            }

            (BorrowKind::Mut { .. }, _, _, BorrowKind::Mut { .. }, _, _) => {
                first_borrow_desc = "first ";
                tcx.cannot_mutably_borrow_multiply(
                    span,
                    &desc_place,
                    &msg_place,
                    issued_span,
                    &msg_borrow,
                    None,
                    Origin::Mir,
                )
            }

            (BorrowKind::Unique, _, _, BorrowKind::Unique, _, _) => {
                first_borrow_desc = "first ";
                tcx.cannot_uniquely_borrow_by_two_closures(
                    span,
                    &desc_place,
                    issued_span,
                    None,
                    Origin::Mir,
                )
            }

            (BorrowKind::Mut { .. }, _, _, BorrowKind::Shallow, _, _)
            | (BorrowKind::Unique, _, _, BorrowKind::Shallow, _, _) => {
                let mut err = tcx.cannot_mutate_in_match_guard(
                    span,
                    issued_span,
                    &desc_place,
                    "mutably borrow",
                    Origin::Mir,
                );
                borrow_spans.var_span_label(
                    &mut err,
                    format!(
                        "borrow occurs due to use of `{}`{}", desc_place, borrow_spans.describe()
                    ),
                );
                err.buffer(&mut self.errors_buffer);

                return;
            }

            (BorrowKind::Unique, _, _, _, _, _) => {
                first_borrow_desc = "first ";
                tcx.cannot_uniquely_borrow_by_one_closure(
                    span,
                    container_name,
                    &desc_place,
                    "",
                    issued_span,
                    "it",
                    "",
                    None,
                    Origin::Mir,
                )
            },

            (BorrowKind::Shared, lft, _, BorrowKind::Unique, _, _) => {
                first_borrow_desc = "first ";
                tcx.cannot_reborrow_already_uniquely_borrowed(
                    span,
                    container_name,
                    &desc_place,
                    "",
                    lft,
                    issued_span,
                    "",
                    None,
                    second_borrow_desc,
                    Origin::Mir,
                )
            }

            (BorrowKind::Mut { .. }, _, lft, BorrowKind::Unique, _, _) => {
                first_borrow_desc = "first ";
                tcx.cannot_reborrow_already_uniquely_borrowed(
                    span,
                    container_name,
                    &desc_place,
                    "",
                    lft,
                    issued_span,
                    "",
                    None,
                    second_borrow_desc,
                    Origin::Mir,
                )
            }

            (BorrowKind::Shared, _, _, BorrowKind::Shared, _, _)
            | (BorrowKind::Shared, _, _, BorrowKind::Shallow, _, _)
            | (BorrowKind::Shallow, _, _, BorrowKind::Mut { .. }, _, _)
            | (BorrowKind::Shallow, _, _, BorrowKind::Unique, _, _)
            | (BorrowKind::Shallow, _, _, BorrowKind::Shared, _, _)
            | (BorrowKind::Shallow, _, _, BorrowKind::Shallow, _, _) => unreachable!(),
        };

        if issued_spans == borrow_spans {
            borrow_spans.var_span_label(
                &mut err,
                format!("borrows occur due to use of `{}`{}", desc_place, borrow_spans.describe()),
            );
        } else {
            let borrow_place = &issued_borrow.borrowed_place;
            let borrow_place_desc = self.describe_place(borrow_place)
                                        .unwrap_or_else(|| "_".to_owned());
            issued_spans.var_span_label(
                &mut err,
                format!(
                    "first borrow occurs due to use of `{}`{}",
                    borrow_place_desc,
                    issued_spans.describe(),
                ),
            );

            borrow_spans.var_span_label(
                &mut err,
                format!(
                    "second borrow occurs due to use of `{}`{}",
                    desc_place,
                    borrow_spans.describe(),
                ),
            );
        }

        if union_type_name != "" {
            err.note(&format!(
                "`{}` is a field of the union `{}`, so it overlaps the field `{}`",
                msg_place, union_type_name, msg_borrow,
            ));
        }

        explanation.add_explanation_to_diagnostic(
            self.infcx.tcx,
            self.mir,
            &mut err,
            first_borrow_desc,
            None,
        );

        err.buffer(&mut self.errors_buffer);
    }

    /// Returns the description of the root place for a conflicting borrow and the full
    /// descriptions of the places that caused the conflict.
    ///
    /// In the simplest case, where there are no unions involved, if a mutable borrow of `x` is
    /// attempted while a shared borrow is live, then this function will return:
    ///
    ///     ("x", "", "")
    ///
    /// In the simple union case, if a mutable borrow of a union field `x.z` is attempted while
    /// a shared borrow of another field `x.y`, then this function will return:
    ///
    ///     ("x", "x.z", "x.y")
    ///
    /// In the more complex union case, where the union is a field of a struct, then if a mutable
    /// borrow of a union field in a struct `x.u.z` is attempted while a shared borrow of
    /// another field `x.u.y`, then this function will return:
    ///
    ///     ("x.u", "x.u.z", "x.u.y")
    ///
    /// This is used when creating error messages like below:
    ///
    /// >  cannot borrow `a.u` (via `a.u.z.c`) as immutable because it is also borrowed as
    /// >  mutable (via `a.u.s.b`) [E0502]
    pub(super) fn describe_place_for_conflicting_borrow(
        &self,
        first_borrowed_place: &Place<'tcx>,
        second_borrowed_place: &Place<'tcx>,
    ) -> (String, String, String, String) {
        // Define a small closure that we can use to check if the type of a place
        // is a union.
        let is_union = |place: &Place<'tcx>| -> bool {
            place.ty(self.mir, self.infcx.tcx).ty
                .ty_adt_def()
                .map(|adt| adt.is_union())
                .unwrap_or(false)
        };

        // Start with an empty tuple, so we can use the functions on `Option` to reduce some
        // code duplication (particularly around returning an empty description in the failure
        // case).
        Some(())
            .filter(|_| {
                // If we have a conflicting borrow of the same place, then we don't want to add
                // an extraneous "via x.y" to our diagnostics, so filter out this case.
                first_borrowed_place != second_borrowed_place
            })
            .and_then(|_| {
                // We're going to want to traverse the first borrowed place to see if we can find
                // field access to a union. If we find that, then we will keep the place of the
                // union being accessed and the field that was being accessed so we can check the
                // second borrowed place for the same union and a access to a different field.
                let mut current = first_borrowed_place;
                while let Place::Projection(box PlaceProjection { base, elem }) = current {
                    match elem {
                        ProjectionElem::Field(field, _) if is_union(base) => {
                            return Some((base, field));
                        },
                        _ => current = base,
                    }
                }
                None
            })
            .and_then(|(target_base, target_field)| {
                // With the place of a union and a field access into it, we traverse the second
                // borrowed place and look for a access to a different field of the same union.
                let mut current = second_borrowed_place;
                while let Place::Projection(box PlaceProjection { base, elem }) = current {
                    match elem {
                        ProjectionElem::Field(field, _) if {
                            is_union(base) && field != target_field && base == target_base
                        } => {
                            let desc_base = self.describe_place(base)
                                .unwrap_or_else(|| "_".to_owned());
                            let desc_first = self.describe_place(first_borrowed_place)
                                .unwrap_or_else(|| "_".to_owned());
                            let desc_second = self.describe_place(second_borrowed_place)
                                .unwrap_or_else(|| "_".to_owned());

                            // Also compute the name of the union type, eg. `Foo` so we
                            // can add a helpful note with it.
                            let ty = base.ty(self.mir, self.infcx.tcx).ty;

                            return Some((desc_base, desc_first, desc_second, ty.to_string()));
                        },
                        _ => current = base,
                    }
                }
                None
            })
            .unwrap_or_else(|| {
                // If we didn't find a field access into a union, or both places match, then
                // only return the description of the first place.
                let desc_place = self.describe_place(first_borrowed_place)
                    .unwrap_or_else(|| "_".to_owned());
                (desc_place, "".to_string(), "".to_string(), "".to_string())
            })
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
        debug!(
            "report_borrowed_value_does_not_live_long_enough(\
             {:?}, {:?}, {:?}, {:?}\
             )",
            context, borrow, place_span, kind
        );

        let drop_span = place_span.1;
        let scope_tree = self.infcx.tcx.region_scope_tree(self.mir_def_id);
        let root_place = self.prefixes(&borrow.borrowed_place, PrefixSet::All)
            .last()
            .unwrap();

        let borrow_spans = self.retrieve_borrow_spans(borrow);
        let borrow_span = borrow_spans.var_or_use();

        let proper_span = match *root_place {
            Place::Base(PlaceBase::Local(local)) => self.mir.local_decls[local].source_info.span,
            _ => drop_span,
        };

        if self.access_place_error_reported
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

        if let StorageDeadOrDrop::Destructor(dropped_ty) =
            self.classify_drop_access_kind(&borrow.borrowed_place)
        {
            // If a borrow of path `B` conflicts with drop of `D` (and
            // we're not in the uninteresting case where `B` is a
            // prefix of `D`), then report this as a more interesting
            // destructor conflict.
            if !borrow.borrowed_place.is_prefix_of(place_span.0) {
                self.report_borrow_conflicts_with_destructor(
                    context, borrow, place_span, kind, dropped_ty,
                );
                return;
            }
        }

        let place_desc = self.describe_place(&borrow.borrowed_place);

        let kind_place = kind.filter(|_| place_desc.is_some()).map(|k| (k, place_span.0));
        let explanation = self.explain_why_borrow_contains_point(context, &borrow, kind_place);

        let err = match (place_desc, explanation) {
            (Some(_), _) if self.is_place_thread_local(root_place) => {
                self.report_thread_local_value_does_not_live_long_enough(drop_span, borrow_span)
            }
            // If the outlives constraint comes from inside the closure,
            // for example:
            //
            // let x = 0;
            // let y = &x;
            // Box::new(|| y) as Box<Fn() -> &'static i32>
            //
            // then just use the normal error. The closure isn't escaping
            // and `move` will not help here.
            (
                Some(ref name),
                BorrowExplanation::MustBeValidFor {
                    category: category @ ConstraintCategory::Return,
                    from_closure: false,
                    ref region_name,
                    span,
                    ..
                },
            )
            | (
                Some(ref name),
                BorrowExplanation::MustBeValidFor {
                    category: category @ ConstraintCategory::CallArgument,
                    from_closure: false,
                    ref region_name,
                    span,
                    ..
                },
            ) if borrow_spans.for_closure() => self.report_escaping_closure_capture(
                borrow_spans.args_or_use(),
                borrow_span,
                region_name,
                category,
                span,
                &format!("`{}`", name),
            ),
            (
                ref name,
                BorrowExplanation::MustBeValidFor {
                    category: ConstraintCategory::Assignment,
                    from_closure: false,
                    region_name: RegionName {
                        source: RegionNameSource::AnonRegionFromUpvar(upvar_span, ref upvar_name),
                        ..
                    },
                    span,
                    ..
                },
            ) => self.report_escaping_data(borrow_span, name, upvar_span, upvar_name, span),
            (Some(name), explanation) => self.report_local_value_does_not_live_long_enough(
                context,
                &name,
                &scope_tree,
                &borrow,
                drop_span,
                borrow_spans,
                explanation,
            ),
            (None, explanation) => self.report_temporary_value_does_not_live_long_enough(
                context,
                &scope_tree,
                &borrow,
                drop_span,
                borrow_spans,
                proper_span,
                explanation,
            ),
        };

        err.buffer(&mut self.errors_buffer);
    }

    fn report_local_value_does_not_live_long_enough(
        &mut self,
        context: Context,
        name: &str,
        scope_tree: &Lrc<ScopeTree>,
        borrow: &BorrowData<'tcx>,
        drop_span: Span,
        borrow_spans: UseSpans,
        explanation: BorrowExplanation,
    ) -> DiagnosticBuilder<'cx> {
        debug!(
            "report_local_value_does_not_live_long_enough(\
             {:?}, {:?}, {:?}, {:?}, {:?}, {:?}\
             )",
            context, name, scope_tree, borrow, drop_span, borrow_spans
        );

        let borrow_span = borrow_spans.var_or_use();
        if let BorrowExplanation::MustBeValidFor {
            category: ConstraintCategory::Return,
            span,
            ref opt_place_desc,
            from_closure: false,
            ..
        } = explanation {
            return self.report_cannot_return_reference_to_local(
                borrow,
                borrow_span,
                span,
                opt_place_desc.as_ref(),
            );
        }

        let mut err = self.infcx.tcx.path_does_not_live_long_enough(
            borrow_span,
            &format!("`{}`", name),
            Origin::Mir,
        );

        if let Some(annotation) = self.annotate_argument_and_return_for_borrow(borrow) {
            let region_name = annotation.emit(self, &mut err);

            err.span_label(
                borrow_span,
                format!("`{}` would have to be valid for `{}`...", name, region_name),
            );

            if let Some(fn_hir_id) = self.infcx.tcx.hir().as_local_hir_id(self.mir_def_id) {
                err.span_label(
                    drop_span,
                    format!(
                        "...but `{}` will be dropped here, when the function `{}` returns",
                        name,
                        self.infcx.tcx.hir().name_by_hir_id(fn_hir_id),
                    ),
                );

                err.note(
                    "functions cannot return a borrow to data owned within the function's scope, \
                     functions can only return borrows to data passed as arguments",
                );
                err.note(
                    "to learn more, visit <https://doc.rust-lang.org/book/ch04-02-\
                     references-and-borrowing.html#dangling-references>",
                );
            } else {
                err.span_label(
                    drop_span,
                    format!("...but `{}` dropped here while still borrowed", name),
                );
            }

            if let BorrowExplanation::MustBeValidFor { .. } = explanation {
            } else {
                explanation.add_explanation_to_diagnostic(
                    self.infcx.tcx,
                    self.mir,
                    &mut err,
                    "",
                    None,
                );
            }
        } else {
            err.span_label(borrow_span, "borrowed value does not live long enough");
            err.span_label(
                drop_span,
                format!("`{}` dropped here while still borrowed", name),
            );

            let within = if borrow_spans.for_generator() {
                " by generator"
            } else {
                ""
            };

            borrow_spans.args_span_label(
                &mut err,
                format!("value captured here{}", within),
            );

            explanation.add_explanation_to_diagnostic(self.infcx.tcx, self.mir, &mut err, "", None);
        }

        err
    }

    fn report_borrow_conflicts_with_destructor(
        &mut self,
        context: Context,
        borrow: &BorrowData<'tcx>,
        (place, drop_span): (&Place<'tcx>, Span),
        kind: Option<WriteKind>,
        dropped_ty: ty::Ty<'tcx>,
    ) {
        debug!(
            "report_borrow_conflicts_with_destructor(\
             {:?}, {:?}, ({:?}, {:?}), {:?}\
             )",
            context, borrow, place, drop_span, kind,
        );

        let borrow_spans = self.retrieve_borrow_spans(borrow);
        let borrow_span = borrow_spans.var_or_use();

        let mut err = self.infcx
            .tcx
            .cannot_borrow_across_destructor(borrow_span, Origin::Mir);

        let what_was_dropped = match self.describe_place(place) {
            Some(name) => format!("`{}`", name.as_str()),
            None => String::from("temporary value"),
        };

        let label = match self.describe_place(&borrow.borrowed_place) {
            Some(borrowed) => format!(
                "here, drop of {D} needs exclusive access to `{B}`, \
                 because the type `{T}` implements the `Drop` trait",
                D = what_was_dropped,
                T = dropped_ty,
                B = borrowed
            ),
            None => format!(
                "here is drop of {D}; whose type `{T}` implements the `Drop` trait",
                D = what_was_dropped,
                T = dropped_ty
            ),
        };
        err.span_label(drop_span, label);

        // Only give this note and suggestion if they could be relevant.
        let explanation =
            self.explain_why_borrow_contains_point(context, borrow, kind.map(|k| (k, place)));
        match explanation {
            BorrowExplanation::UsedLater { .. }
            | BorrowExplanation::UsedLaterWhenDropped { .. } => {
                err.note("consider using a `let` binding to create a longer lived value");
            }
            _ => {}
        }

        explanation.add_explanation_to_diagnostic(self.infcx.tcx, self.mir, &mut err, "", None);

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

        let mut err = self.infcx
            .tcx
            .thread_local_value_does_not_live_long_enough(borrow_span, Origin::Mir);

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
        explanation: BorrowExplanation,
    ) -> DiagnosticBuilder<'cx> {
        debug!(
            "report_temporary_value_does_not_live_long_enough(\
             {:?}, {:?}, {:?}, {:?}, {:?}\
             )",
            context, scope_tree, borrow, drop_span, proper_span
        );

        if let BorrowExplanation::MustBeValidFor {
            category: ConstraintCategory::Return,
            span,
            from_closure: false,
            ..
        } = explanation {
            return self.report_cannot_return_reference_to_local(
                borrow,
                proper_span,
                span,
                None,
            );
        }

        let tcx = self.infcx.tcx;
        let mut err = tcx.temporary_value_borrowed_for_too_long(proper_span, Origin::Mir);
        err.span_label(
            proper_span,
            "creates a temporary which is freed while still in use",
        );
        err.span_label(
            drop_span,
            "temporary value is freed at the end of this statement",
        );

        match explanation {
            BorrowExplanation::UsedLater(..)
            | BorrowExplanation::UsedLaterInLoop(..)
            | BorrowExplanation::UsedLaterWhenDropped { .. } => {
                // Only give this note and suggestion if it could be relevant.
                err.note("consider using a `let` binding to create a longer lived value");
            }
            _ => {}
        }
        explanation.add_explanation_to_diagnostic(self.infcx.tcx, self.mir, &mut err, "", None);

        let within = if borrow_spans.for_generator() {
            " by generator"
        } else {
            ""
        };

        borrow_spans.args_span_label(
            &mut err,
            format!("value captured here{}", within),
        );

        err
    }

    fn report_cannot_return_reference_to_local(
        &self,
        borrow: &BorrowData<'tcx>,
        borrow_span: Span,
        return_span: Span,
        opt_place_desc: Option<&String>,
    ) -> DiagnosticBuilder<'cx> {
        let tcx = self.infcx.tcx;

        // FIXME use a better heuristic than Spans
        let reference_desc = if return_span == self.mir.source_info(borrow.reserve_location).span {
            "reference to"
        } else {
            "value referencing"
        };

        let (place_desc, note) = if let Some(place_desc) = opt_place_desc {
            let local_kind = match borrow.borrowed_place {
                Place::Base(PlaceBase::Local(local)) => {
                    match self.mir.local_kind(local) {
                        LocalKind::ReturnPointer
                        | LocalKind::Temp => bug!("temporary or return pointer with a name"),
                        LocalKind::Var => "local variable ",
                        LocalKind::Arg
                        if !self.mir.upvar_decls.is_empty()
                            && local == Local::new(1) => {
                            "variable captured by `move` "
                        }
                        LocalKind::Arg => {
                            "function parameter "
                        }
                    }
                }
                _ => "local data ",
            };
            (
                format!("{}`{}`", local_kind, place_desc),
                format!("`{}` is borrowed here", place_desc),
            )
        } else {
            let root_place = self.prefixes(&borrow.borrowed_place, PrefixSet::All)
                .last()
                .unwrap();
            let local = if let Place::Base(PlaceBase::Local(local)) = *root_place {
                local
            } else {
                bug!("report_cannot_return_reference_to_local: not a local")
            };
            match self.mir.local_kind(local) {
                LocalKind::ReturnPointer | LocalKind::Temp => {
                    (
                        "temporary value".to_string(),
                        "temporary value created here".to_string(),
                    )
                }
                LocalKind::Arg => {
                    (
                        "function parameter".to_string(),
                        "function parameter borrowed here".to_string(),
                    )
                },
                LocalKind::Var => bug!("local variable without a name"),
            }
        };

        let mut err = tcx.cannot_return_reference_to_local(
            return_span,
            reference_desc,
            &place_desc,
            Origin::Mir,
        );

        if return_span != borrow_span {
            err.span_label(borrow_span, note);
        }

        err
    }

    fn report_escaping_closure_capture(
        &mut self,
        args_span: Span,
        var_span: Span,
        fr_name: &RegionName,
        category: ConstraintCategory,
        constraint_span: Span,
        captured_var: &str,
    ) -> DiagnosticBuilder<'cx> {
        let tcx = self.infcx.tcx;

        let mut err = tcx.cannot_capture_in_long_lived_closure(
            args_span,
            captured_var,
            var_span,
          Origin::Mir,
        );

        let suggestion = match tcx.sess.source_map().span_to_snippet(args_span) {
            Ok(string) => format!("move {}", string),
            Err(_) => "move |<args>| <body>".to_string()
        };

        err.span_suggestion(
            args_span,
            &format!("to force the closure to take ownership of {} (and any \
                      other referenced variables), use the `move` keyword",
                      captured_var),
            suggestion,
            Applicability::MachineApplicable,
        );

        match category {
            ConstraintCategory::Return => {
                err.span_note(constraint_span, "closure is returned here");
            }
            ConstraintCategory::CallArgument => {
                fr_name.highlight_region_name(&mut err);
                err.span_note(
                    constraint_span,
                    &format!("function requires argument type to outlive `{}`", fr_name),
                );
            }
            _ => bug!("report_escaping_closure_capture called with unexpected constraint \
                       category: `{:?}`", category),
        }
        err
    }

    fn report_escaping_data(
        &mut self,
        borrow_span: Span,
        name: &Option<String>,
        upvar_span: Span,
        upvar_name: &str,
        escape_span: Span,
    ) -> DiagnosticBuilder<'cx> {
        let tcx = self.infcx.tcx;

        let escapes_from = if tcx.is_closure(self.mir_def_id) {
            let tables = tcx.typeck_tables_of(self.mir_def_id);
            let mir_hir_id = tcx.hir().def_index_to_hir_id(self.mir_def_id.index);
            match tables.node_type(mir_hir_id).sty {
                ty::Closure(..) => "closure",
                ty::Generator(..) => "generator",
                _ => bug!("Closure body doesn't have a closure or generator type"),
            }
        } else {
            "function"
        };

        let mut err = tcx.borrowed_data_escapes_closure(escape_span, escapes_from, Origin::Mir);

        err.span_label(
            upvar_span,
            format!(
                "`{}` is declared here, outside of the {} body",
                upvar_name, escapes_from
            ),
        );

        err.span_label(
            borrow_span,
            format!(
                "borrow is only valid in the {} body",
                escapes_from
            ),
        );

        if let Some(name) = name {
            err.span_label(
                escape_span,
                format!("reference to `{}` escapes the {} body here", name, escapes_from),
            );
        } else {
            err.span_label(
                escape_span,
                format!("reference escapes the {} body here", escapes_from),
            );
        }

        err
    }

    fn get_moved_indexes(&mut self, context: Context, mpi: MovePathIndex) -> Vec<MoveSite> {
        let mir = self.mir;

        let mut stack = Vec::new();
        stack.extend(mir.predecessor_locations(context.loc).map(|predecessor| {
            let is_back_edge = context.loc.dominates(predecessor, &self.dominators);
            (predecessor, is_back_edge)
        }));

        let mut visited = FxHashSet::default();
        let mut result = vec![];

        'dfs: while let Some((location, is_back_edge)) = stack.pop() {
            debug!(
                "report_use_of_moved_or_uninitialized: (current_location={:?}, back_edge={})",
                location, is_back_edge
            );

            if !visited.insert(location) {
                continue;
            }

            // check for moves
            let stmt_kind = mir[location.block]
                .statements
                .get(location.statement_index)
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

                for moi in &self.move_data.loc_map[location] {
                    debug!("report_use_of_moved_or_uninitialized: moi={:?}", moi);
                    if mpis.contains(&self.move_data.moves[*moi].path) {
                        debug!("report_use_of_moved_or_uninitialized: found");
                        result.push(MoveSite {
                            moi: *moi,
                            traversed_back_edge: is_back_edge,
                        });

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
            drop_flag_effects::for_location_inits(
                self.infcx.tcx,
                self.mir,
                self.move_data,
                location,
                |m| {
                    if m == mpi {
                        any_match = true;
                    }
                },
            );
            if any_match {
                continue 'dfs;
            }

            stack.extend(mir.predecessor_locations(location).map(|predecessor| {
                let back_edge = location.dominates(predecessor, &self.dominators);
                (predecessor, is_back_edge || back_edge)
            }));
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
        if loan.kind == BorrowKind::Shallow {
            let mut err = tcx.cannot_mutate_in_match_guard(
                span,
                loan_span,
                &self.describe_place(place).unwrap_or_else(|| "_".to_owned()),
                "assign",
                Origin::Mir,
            );
            loan_spans.var_span_label(
                &mut err,
                format!("borrow occurs due to use{}", loan_spans.describe()),
            );

            err.buffer(&mut self.errors_buffer);

            return;
        }

        let mut err = tcx.cannot_assign_to_borrowed(
            span,
            loan_span,
            &self.describe_place(place).unwrap_or_else(|| "_".to_owned()),
            Origin::Mir,
        );

        loan_spans.var_span_label(
            &mut err,
            format!("borrow occurs due to use{}", loan_spans.describe()),
        );

        self.explain_why_borrow_contains_point(context, loan, None)
            .add_explanation_to_diagnostic(self.infcx.tcx, self.mir, &mut err, "", None);

        err.buffer(&mut self.errors_buffer);
    }

    /// Reports an illegal reassignment; for example, an assignment to
    /// (part of) a non-`mut` local that occurs potentially after that
    /// local has already been initialized. `place` is the path being
    /// assigned; `err_place` is a place providing a reason why
    /// `place` is not mutable (e.g., the non-`mut` local `x` in an
    /// assignment to `x.f`).
    pub(super) fn report_illegal_reassignment(
        &mut self,
        _context: Context,
        (place, span): (&Place<'tcx>, Span),
        assigned_span: Span,
        err_place: &Place<'tcx>,
    ) {
        let (from_arg, local_decl) = if let Place::Base(PlaceBase::Local(local)) = *err_place {
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
                    err.span_suggestion(
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

/// Which case a StorageDeadOrDrop is for.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum StorageDeadOrDrop<'tcx> {
    LocalStorageDead,
    BoxedStorageDead,
    Destructor(ty::Ty<'tcx>),
}

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {

    /// Adds a suggestion when a closure is invoked twice with a moved variable or when a closure
    /// is moved after being invoked.
    ///
    /// ```text
    /// note: closure cannot be invoked more than once because it moves the variable `dict` out of
    ///       its environment
    ///   --> $DIR/issue-42065.rs:16:29
    ///    |
    /// LL |         for (key, value) in dict {
    ///    |                             ^^^^
    /// ```
    pub(super) fn add_moved_or_invoked_closure_note(
        &self,
        location: Location,
        place: &Place<'tcx>,
        diag: &mut DiagnosticBuilder<'_>,
    ) {
        debug!("add_moved_or_invoked_closure_note: location={:?} place={:?}", location, place);
        let mut target = place.local();
        for stmt in &self.mir[location.block].statements[location.statement_index..] {
            debug!("add_moved_or_invoked_closure_note: stmt={:?} target={:?}", stmt, target);
            if let StatementKind::Assign(into, box Rvalue::Use(from)) = &stmt.kind {
                debug!("add_fnonce_closure_note: into={:?} from={:?}", into, from);
                match from {
                    Operand::Copy(ref place) |
                    Operand::Move(ref place) if target == place.local() =>
                        target = into.local(),
                    _ => {},
                }
            }
        }

        // Check if we are attempting to call a closure after it has been invoked.
        let terminator = self.mir[location.block].terminator();
        debug!("add_moved_or_invoked_closure_note: terminator={:?}", terminator);
        if let TerminatorKind::Call {
            func: Operand::Constant(box Constant {
                literal: ty::Const {
                    ty: &ty::TyS { sty: ty::TyKind::FnDef(id, _), ..  },
                    ..
                },
                ..
            }),
            args,
            ..
        } = &terminator.kind {
            debug!("add_moved_or_invoked_closure_note: id={:?}", id);
            if self.infcx.tcx.parent(id) == self.infcx.tcx.lang_items().fn_once_trait() {
                let closure = match args.first() {
                    Some(Operand::Copy(ref place)) |
                    Some(Operand::Move(ref place)) if target == place.local() =>
                        place.local().unwrap(),
                    _ => return,
                };

                debug!("add_moved_or_invoked_closure_note: closure={:?}", closure);
                if let ty::TyKind::Closure(did, _) = self.mir.local_decls[closure].ty.sty {
                    let hir_id = self.infcx.tcx.hir().as_local_hir_id(did).unwrap();

                    if let Some((span, name)) = self.infcx.tcx.typeck_tables_of(did)
                        .closure_kind_origins()
                        .get(hir_id)
                    {
                        diag.span_note(
                            *span,
                            &format!(
                                "closure cannot be invoked more than once because it moves the \
                                 variable `{}` out of its environment",
                                name,
                            ),
                        );
                        return;
                    }
                }
            }
        }

        // Check if we are just moving a closure after it has been invoked.
        if let Some(target) = target {
            if let ty::TyKind::Closure(did, _) = self.mir.local_decls[target].ty.sty {
                let hir_id = self.infcx.tcx.hir().as_local_hir_id(did).unwrap();

                if let Some((span, name)) = self.infcx.tcx.typeck_tables_of(did)
                    .closure_kind_origins()
                    .get(hir_id)
                {
                    diag.span_note(
                        *span,
                        &format!(
                            "closure cannot be moved more than once as it is not `Copy` due to \
                             moving the variable `{}` out of its environment",
                             name
                        ),
                    );
                }
            }
        }
    }

    /// End-user visible description of `place` if one can be found. If the
    /// place is a temporary for instance, None will be returned.
    pub(super) fn describe_place(&self, place: &Place<'tcx>) -> Option<String> {
        self.describe_place_with_options(place, IncludingDowncast(false))
    }

    /// End-user visible description of `place` if one can be found. If the
    /// place is a temporary for instance, None will be returned.
    /// `IncludingDowncast` parameter makes the function return `Err` if `ProjectionElem` is
    /// `Downcast` and `IncludingDowncast` is true
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

    /// Appends end-user visible description of `place` to `buf`.
    fn append_place_to_string(
        &self,
        place: &Place<'tcx>,
        buf: &mut String,
        mut autoderef: bool,
        including_downcast: &IncludingDowncast,
    ) -> Result<(), ()> {
        match *place {
            Place::Base(PlaceBase::Local(local)) => {
                self.append_local_to_string(local, buf)?;
            }
            Place::Base(PlaceBase::Static(box Static{ kind: StaticKind::Promoted(_), .. })) => {
                buf.push_str("promoted");
            }
            Place::Base(PlaceBase::Static(box Static{ kind: StaticKind::Static(def_id), .. })) => {
                buf.push_str(&self.infcx.tcx.item_name(def_id).to_string());
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
                            } else if let Place::Base(PlaceBase::Local(local)) = proj.base {
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
                            buf.push_str("_");
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

    /// Appends end-user visible description of the `local` place to `buf`. If `local` doesn't have
    /// a name, then `Err` is returned
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

    /// End-user visible description of the `field`nth field of `base`
    fn describe_field(&self, base: &Place<'tcx>, field: Field) -> String {
        match *base {
            Place::Base(PlaceBase::Local(local)) => {
                let local = &self.mir.local_decls[local];
                self.describe_field_from_ty(&local.ty, field, None)
            }
            Place::Base(PlaceBase::Static(ref static_)) =>
                self.describe_field_from_ty(&static_.ty, field, None),
            Place::Projection(ref proj) => match proj.elem {
                ProjectionElem::Deref => self.describe_field(&proj.base, field),
                ProjectionElem::Downcast(_, variant_index) => {
                    let base_ty = base.ty(self.mir, self.infcx.tcx).ty;
                    self.describe_field_from_ty(&base_ty, field, Some(variant_index))
                }
                ProjectionElem::Field(_, field_type) => {
                    self.describe_field_from_ty(&field_type, field, None)
                }
                ProjectionElem::Index(..)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. } => {
                    self.describe_field(&proj.base, field)
                }
            },
        }
    }

    /// End-user visible description of the `field_index`nth field of `ty`
    fn describe_field_from_ty(
        &self,
        ty: &ty::Ty<'_>,
        field: Field,
        variant_index: Option<VariantIdx>
    ) -> String {
        if ty.is_box() {
            // If the type is a box, the field is described from the boxed type
            self.describe_field_from_ty(&ty.boxed_ty(), field, variant_index)
        } else {
            match ty.sty {
                ty::Adt(def, _) => {
                    let variant = if let Some(idx) = variant_index {
                        assert!(def.is_enum());
                        &def.variants[idx]
                    } else {
                        def.non_enum_variant()
                    };
                    variant.fields[field.index()]
                        .ident
                        .to_string()
                },
                ty::Tuple(_) => field.index().to_string(),
                ty::Ref(_, ty, _) | ty::RawPtr(ty::TypeAndMut { ty, .. }) => {
                    self.describe_field_from_ty(&ty, field, variant_index)
                }
                ty::Array(ty, _) | ty::Slice(ty) =>
                    self.describe_field_from_ty(&ty, field, variant_index),
                ty::Closure(def_id, _) | ty::Generator(def_id, _, _) => {
                    // Convert the def-id into a node-id. node-ids are only valid for
                    // the local code in the current crate, so this returns an `Option` in case
                    // the closure comes from another crate. But in that case we wouldn't
                    // be borrowck'ing it, so we can just unwrap:
                    let hir_id = self.infcx.tcx.hir().as_local_hir_id(def_id).unwrap();
                    let freevar = self.infcx
                        .tcx
                        .with_freevars(hir_id, |fv| fv[field.index()]);

                    self.infcx.tcx.hir().name(freevar.var_id()).to_string()
                }
                _ => {
                    // Might need a revision when the fields in trait RFC is implemented
                    // (https://github.com/rust-lang/rfcs/pull/1546)
                    bug!(
                        "End-user description not implemented for field access on `{:?}`",
                        ty
                    );
                }
            }
        }
    }

    /// Checks if a place is a thread-local static.
    pub fn is_place_thread_local(&self, place: &Place<'tcx>) -> bool {
        if let Place::Base(
            PlaceBase::Static(box Static{ kind: StaticKind::Static(def_id), .. })
        ) = place {
            let attrs = self.infcx.tcx.get_attrs(*def_id);
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

    fn classify_drop_access_kind(&self, place: &Place<'tcx>) -> StorageDeadOrDrop<'tcx> {
        let tcx = self.infcx.tcx;
        match place {
            Place::Base(PlaceBase::Local(_)) |
            Place::Base(PlaceBase::Static(_)) => {
                StorageDeadOrDrop::LocalStorageDead
            }
            Place::Projection(box PlaceProjection { base, elem }) => {
                let base_access = self.classify_drop_access_kind(base);
                match elem {
                    ProjectionElem::Deref => match base_access {
                        StorageDeadOrDrop::LocalStorageDead
                        | StorageDeadOrDrop::BoxedStorageDead => {
                            assert!(
                                base.ty(self.mir, tcx).ty.is_box(),
                                "Drop of value behind a reference or raw pointer"
                            );
                            StorageDeadOrDrop::BoxedStorageDead
                        }
                        StorageDeadOrDrop::Destructor(_) => base_access,
                    },
                    ProjectionElem::Field(..) | ProjectionElem::Downcast(..) => {
                        let base_ty = base.ty(self.mir, tcx).ty;
                        match base_ty.sty {
                            ty::Adt(def, _) if def.has_dtor(tcx) => {
                                // Report the outermost adt with a destructor
                                match base_access {
                                    StorageDeadOrDrop::Destructor(_) => base_access,
                                    StorageDeadOrDrop::LocalStorageDead
                                    | StorageDeadOrDrop::BoxedStorageDead => {
                                        StorageDeadOrDrop::Destructor(base_ty)
                                    }
                                }
                            }
                            _ => base_access,
                        }
                    }

                    ProjectionElem::ConstantIndex { .. }
                    | ProjectionElem::Subslice { .. }
                    | ProjectionElem::Index(_) => base_access,
                }
            }
        }
    }

    /// Annotate argument and return type of function and closure with (synthesized) lifetime for
    /// borrow of local value that does not live long enough.
    fn annotate_argument_and_return_for_borrow(
        &self,
        borrow: &BorrowData<'tcx>,
    ) -> Option<AnnotatedBorrowFnSignature<'tcx>> {
        // Define a fallback for when we can't match a closure.
        let fallback = || {
            let is_closure = self.infcx.tcx.is_closure(self.mir_def_id);
            if is_closure {
                None
            } else {
                let ty = self.infcx.tcx.type_of(self.mir_def_id);
                match ty.sty {
                    ty::TyKind::FnDef(_, _) | ty::TyKind::FnPtr(_) => self.annotate_fn_sig(
                        self.mir_def_id,
                        self.infcx.tcx.fn_sig(self.mir_def_id),
                    ),
                    _ => None,
                }
            }
        };

        // In order to determine whether we need to annotate, we need to check whether the reserve
        // place was an assignment into a temporary.
        //
        // If it was, we check whether or not that temporary is eventually assigned into the return
        // place. If it was, we can add annotations about the function's return type and arguments
        // and it'll make sense.
        let location = borrow.reserve_location;
        debug!(
            "annotate_argument_and_return_for_borrow: location={:?}",
            location
        );
        if let Some(&Statement { kind: StatementKind::Assign(ref reservation, _), ..})
             = &self.mir[location.block].statements.get(location.statement_index)
        {
            debug!(
                "annotate_argument_and_return_for_borrow: reservation={:?}",
                reservation
            );
            // Check that the initial assignment of the reserve location is into a temporary.
            let mut target = *match reservation {
                Place::Base(PlaceBase::Local(local))
                    if self.mir.local_kind(*local) == LocalKind::Temp => local,
                _ => return None,
            };

            // Next, look through the rest of the block, checking if we are assigning the
            // `target` (that is, the place that contains our borrow) to anything.
            let mut annotated_closure = None;
            for stmt in &self.mir[location.block].statements[location.statement_index + 1..] {
                debug!(
                    "annotate_argument_and_return_for_borrow: target={:?} stmt={:?}",
                    target, stmt
                );
                if let StatementKind::Assign(
                    Place::Base(PlaceBase::Local(assigned_to)),
                    box rvalue
                ) = &stmt.kind {
                    debug!(
                        "annotate_argument_and_return_for_borrow: assigned_to={:?} \
                         rvalue={:?}",
                        assigned_to, rvalue
                    );
                    // Check if our `target` was captured by a closure.
                    if let Rvalue::Aggregate(
                        box AggregateKind::Closure(def_id, substs),
                        operands,
                    ) = rvalue
                    {
                        for operand in operands {
                            let assigned_from = match operand {
                                Operand::Copy(assigned_from) | Operand::Move(assigned_from) => {
                                    assigned_from
                                }
                                _ => continue,
                            };
                            debug!(
                                "annotate_argument_and_return_for_borrow: assigned_from={:?}",
                                assigned_from
                            );

                            // Find the local from the operand.
                            let assigned_from_local = match assigned_from.local() {
                                Some(local) => local,
                                None => continue,
                            };

                            if assigned_from_local != target {
                                continue;
                            }

                            // If a closure captured our `target` and then assigned
                            // into a place then we should annotate the closure in
                            // case it ends up being assigned into the return place.
                            annotated_closure = self.annotate_fn_sig(
                                *def_id,
                                self.infcx.closure_sig(*def_id, *substs),
                            );
                            debug!(
                                "annotate_argument_and_return_for_borrow: \
                                 annotated_closure={:?} assigned_from_local={:?} \
                                 assigned_to={:?}",
                                annotated_closure, assigned_from_local, assigned_to
                            );

                            if *assigned_to == mir::RETURN_PLACE {
                                // If it was assigned directly into the return place, then
                                // return now.
                                return annotated_closure;
                            } else {
                                // Otherwise, update the target.
                                target = *assigned_to;
                            }
                        }

                        // If none of our closure's operands matched, then skip to the next
                        // statement.
                        continue;
                    }

                    // Otherwise, look at other types of assignment.
                    let assigned_from = match rvalue {
                        Rvalue::Ref(_, _, assigned_from) => assigned_from,
                        Rvalue::Use(operand) => match operand {
                            Operand::Copy(assigned_from) | Operand::Move(assigned_from) => {
                                assigned_from
                            }
                            _ => continue,
                        },
                        _ => continue,
                    };
                    debug!(
                        "annotate_argument_and_return_for_borrow: \
                         assigned_from={:?}",
                        assigned_from,
                    );

                    // Find the local from the rvalue.
                    let assigned_from_local = match assigned_from.local() {
                        Some(local) => local,
                        None => continue,
                    };
                    debug!(
                        "annotate_argument_and_return_for_borrow: \
                         assigned_from_local={:?}",
                        assigned_from_local,
                    );

                    // Check if our local matches the target - if so, we've assigned our
                    // borrow to a new place.
                    if assigned_from_local != target {
                        continue;
                    }

                    // If we assigned our `target` into a new place, then we should
                    // check if it was the return place.
                    debug!(
                        "annotate_argument_and_return_for_borrow: \
                         assigned_from_local={:?} assigned_to={:?}",
                        assigned_from_local, assigned_to
                    );
                    if *assigned_to == mir::RETURN_PLACE {
                        // If it was then return the annotated closure if there was one,
                        // else, annotate this function.
                        return annotated_closure.or_else(fallback);
                    }

                    // If we didn't assign into the return place, then we just update
                    // the target.
                    target = *assigned_to;
                }
            }

            // Check the terminator if we didn't find anything in the statements.
            let terminator = &self.mir[location.block].terminator();
            debug!(
                "annotate_argument_and_return_for_borrow: target={:?} terminator={:?}",
                target, terminator
            );
            if let TerminatorKind::Call {
                destination: Some((Place::Base(PlaceBase::Local(assigned_to)), _)),
                args,
                ..
            } = &terminator.kind
            {
                debug!(
                    "annotate_argument_and_return_for_borrow: assigned_to={:?} args={:?}",
                    assigned_to, args
                );
                for operand in args {
                    let assigned_from = match operand {
                        Operand::Copy(assigned_from) | Operand::Move(assigned_from) => {
                            assigned_from
                        }
                        _ => continue,
                    };
                    debug!(
                        "annotate_argument_and_return_for_borrow: assigned_from={:?}",
                        assigned_from,
                    );

                    if let Some(assigned_from_local) = assigned_from.local() {
                        debug!(
                            "annotate_argument_and_return_for_borrow: assigned_from_local={:?}",
                            assigned_from_local,
                        );

                        if *assigned_to == mir::RETURN_PLACE && assigned_from_local == target {
                            return annotated_closure.or_else(fallback);
                        }
                    }
                }
            }
        }

        // If we haven't found an assignment into the return place, then we need not add
        // any annotations.
        debug!("annotate_argument_and_return_for_borrow: none found");
        None
    }

    /// Annotate the first argument and return type of a function signature if they are
    /// references.
    fn annotate_fn_sig(
        &self,
        did: DefId,
        sig: ty::PolyFnSig<'tcx>,
    ) -> Option<AnnotatedBorrowFnSignature<'tcx>> {
        debug!("annotate_fn_sig: did={:?} sig={:?}", did, sig);
        let is_closure = self.infcx.tcx.is_closure(did);
        let fn_hir_id = self.infcx.tcx.hir().as_local_hir_id(did)?;
        let fn_decl = self.infcx.tcx.hir().fn_decl_by_hir_id(fn_hir_id)?;

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
                            if let hir::TyKind::Rptr(lifetime, _) = &fn_decl.inputs[index].node {
                                // With access to the lifetime, we can get
                                // the span of it.
                                arguments.push((*argument, lifetime.span));
                            } else {
                                bug!("ty type is a ref but hir type is not");
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
            }
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
            }
            ty::TyKind::Ref(_, _, _) => {
                // This is also case 2 from above but for functions, return type is still an
                // anonymous reference so we select the first argument.
                let argument_span = fn_decl.inputs.first()?.span;
                let argument_ty = sig.inputs().skip_binder().first()?;

                let return_span = fn_decl.output.span();
                let return_ty = *sig.output().skip_binder();

                // We expect the first argument to be a reference.
                match argument_ty.sty {
                    ty::TyKind::Ref(_, _, _) => {}
                    _ => return None,
                }

                Some(AnnotatedBorrowFnSignature::AnonymousFunction {
                    argument_ty,
                    argument_span,
                    return_ty,
                    return_span,
                })
            }
            _ => {
                // This is case 3 from above, return type is not a reference so don't highlight
                // anything.
                None
            }
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
    },
}

impl<'tcx> AnnotatedBorrowFnSignature<'tcx> {
    /// Annotate the provided diagnostic with information about borrow from the fn signature that
    /// helps explain.
    fn emit(
        &self,
        cx: &mut MirBorrowckCtxt<'_, '_, 'tcx>,
        diag: &mut DiagnosticBuilder<'_>,
    ) -> String {
        match self {
            AnnotatedBorrowFnSignature::Closure {
                argument_ty,
                argument_span,
            } => {
                diag.span_label(
                    *argument_span,
                    format!("has type `{}`", cx.get_name_for_ty(argument_ty, 0)),
                );

                cx.get_region_name_for_ty(argument_ty, 0)
            }
            AnnotatedBorrowFnSignature::AnonymousFunction {
                argument_ty,
                argument_span,
                return_ty,
                return_span,
            } => {
                let argument_ty_name = cx.get_name_for_ty(argument_ty, 0);
                diag.span_label(*argument_span, format!("has type `{}`", argument_ty_name));

                let return_ty_name = cx.get_name_for_ty(return_ty, 0);
                let types_equal = return_ty_name == argument_ty_name;
                diag.span_label(
                    *return_span,
                    format!(
                        "{}has type `{}`",
                        if types_equal { "also " } else { "" },
                        return_ty_name,
                    ),
                );

                diag.note(
                    "argument and return type have the same lifetime due to lifetime elision rules",
                );
                diag.note(
                    "to learn more, visit <https://doc.rust-lang.org/book/ch10-03-\
                     lifetime-syntax.html#lifetime-elision>",
                );

                cx.get_region_name_for_ty(return_ty, 0)
            }
            AnnotatedBorrowFnSignature::NamedFunction {
                arguments,
                return_ty,
                return_span,
            } => {
                // Region of return type and arguments checked to be the same earlier.
                let region_name = cx.get_region_name_for_ty(return_ty, 0);
                for (_, argument_span) in arguments {
                    diag.span_label(*argument_span, format!("has lifetime `{}`", region_name));
                }

                diag.span_label(
                    *return_span,
                    format!("also has lifetime `{}`", region_name,),
                );

                diag.help(&format!(
                    "use data from the highlighted arguments which match the `{}` lifetime of \
                     the return type",
                    region_name,
                ));

                region_name
            }
        }
    }
}

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    /// Return the name of the provided `Ty` (that must be a reference) with a synthesized lifetime
    /// name where required.
    fn get_name_for_ty(&self, ty: ty::Ty<'tcx>, counter: usize) -> String {
        let mut s = String::new();
        let mut printer = ty::print::FmtPrinter::new(self.infcx.tcx, &mut s, Namespace::TypeNS);

        // We need to add synthesized lifetimes where appropriate. We do
        // this by hooking into the pretty printer and telling it to label the
        // lifetimes without names with the value `'0`.
        match ty.sty {
            ty::TyKind::Ref(ty::RegionKind::ReLateBound(_, br), _, _)
            | ty::TyKind::Ref(
                ty::RegionKind::RePlaceholder(ty::PlaceholderRegion { name: br, .. }),
                _,
                _,
            ) => printer.region_highlight_mode.highlighting_bound_region(*br, counter),
            _ => {}
        }

        let _ = ty.print(printer);
        s
    }

    /// Returns the name of the provided `Ty` (that must be a reference)'s region with a
    /// synthesized lifetime name where required.
    fn get_region_name_for_ty(&self, ty: ty::Ty<'tcx>, counter: usize) -> String {
        let mut s = String::new();
        let mut printer = ty::print::FmtPrinter::new(self.infcx.tcx, &mut s, Namespace::TypeNS);

        let region = match ty.sty {
            ty::TyKind::Ref(region, _, _) => {
                match region {
                    ty::RegionKind::ReLateBound(_, br)
                    | ty::RegionKind::RePlaceholder(ty::PlaceholderRegion { name: br, .. }) => {
                        printer.region_highlight_mode.highlighting_bound_region(*br, counter)
                    }
                    _ => {}
                }

                region
            }
            _ => bug!("ty for annotation of borrow region is not a reference"),
        };

        let _ = region.print(printer);
        s
    }
}

// The span(s) associated to a use of a place.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum UseSpans {
    // The access is caused by capturing a variable for a closure.
    ClosureUse {
        // This is true if the captured variable was from a generator.
        is_generator: bool,
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
    pub(super) fn args_span_label(
        self,
        err: &mut DiagnosticBuilder<'_>,
        message: impl Into<String>,
    ) {
        if let UseSpans::ClosureUse { args_span, .. } = self {
            err.span_label(args_span, message);
        }
    }

    // Add a span label to the use of the captured variable, if it exists.
    pub(super) fn var_span_label(
        self,
        err: &mut DiagnosticBuilder<'_>,
        message: impl Into<String>,
    ) {
        if let UseSpans::ClosureUse { var_span, .. } = self {
            err.span_label(var_span, message);
        }
    }

    /// Returns `false` if this place is not used in a closure.
    fn for_closure(&self) -> bool {
        match *self {
            UseSpans::ClosureUse { is_generator, .. } => !is_generator,
            _ => false,
        }
    }

    /// Returns `false` if this place is not used in a generator.
    fn for_generator(&self) -> bool {
        match *self {
            UseSpans::ClosureUse { is_generator, .. } => is_generator,
            _ => false,
        }
    }

    /// Describe the span associated with a use of a place.
    fn describe(&self) -> String {
        match *self {
            UseSpans::ClosureUse { is_generator, .. } => if is_generator {
                " in generator".to_string()
            } else {
                " in closure".to_string()
            },
            _ => "".to_string(),
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

        let stmt = match self.mir[location.block].statements.get(location.statement_index) {
            Some(stmt) => stmt,
            None => return OtherUse(self.mir.source_info(location).span),
        };

        debug!("move_spans: moved_place={:?} location={:?} stmt={:?}", moved_place, location, stmt);
        if let  StatementKind::Assign(
            _,
            box Rvalue::Aggregate(ref kind, ref places)
        ) = stmt.kind {
            let (def_id, is_generator) = match kind {
                box AggregateKind::Closure(def_id, _) => (def_id, false),
                box AggregateKind::Generator(def_id, _, _) => (def_id, true),
                _ => return OtherUse(stmt.source_info.span),
            };

            debug!(
                "move_spans: def_id={:?} is_generator={:?} places={:?}",
                def_id, is_generator, places
            );
            if let Some((args_span, var_span)) = self.closure_span(*def_id, moved_place, places) {
                return ClosureUse {
                    is_generator,
                    args_span,
                    var_span,
                };
            }
        }

        OtherUse(stmt.source_info.span)
    }

    /// Finds the span of arguments of a closure (within `maybe_closure_span`)
    /// and its usage of the local assigned at `location`.
    /// This is done by searching in statements succeeding `location`
    /// and originating from `maybe_closure_span`.
    pub(super) fn borrow_spans(&self, use_span: Span, location: Location) -> UseSpans {
        use self::UseSpans::*;
        debug!("borrow_spans: use_span={:?} location={:?}", use_span, location);

        let target = match self.mir[location.block]
            .statements
            .get(location.statement_index)
        {
            Some(&Statement {
                kind: StatementKind::Assign(Place::Base(PlaceBase::Local(local)), _),
                ..
            }) => local,
            _ => return OtherUse(use_span),
        };

        if self.mir.local_kind(target) != LocalKind::Temp {
            // operands are always temporaries.
            return OtherUse(use_span);
        }

        for stmt in &self.mir[location.block].statements[location.statement_index + 1..] {
            if let StatementKind::Assign(
                _, box Rvalue::Aggregate(ref kind, ref places)
            ) = stmt.kind {
                let (def_id, is_generator) = match kind {
                    box AggregateKind::Closure(def_id, _) => (def_id, false),
                    box AggregateKind::Generator(def_id, _, _) => (def_id, true),
                    _ => continue,
                };

                debug!(
                    "borrow_spans: def_id={:?} is_generator={:?} places={:?}",
                    def_id, is_generator, places
                );
                if let Some((args_span, var_span)) = self.closure_span(
                    *def_id, &Place::Base(PlaceBase::Local(target)), places
                ) {
                    return ClosureUse {
                        is_generator,
                        args_span,
                        var_span,
                    };
                } else {
                    return OtherUse(use_span);
                }
            }

            if use_span != stmt.source_info.span {
                break;
            }
        }

        OtherUse(use_span)
    }

    /// Finds the span of a captured variable within a closure or generator.
    fn closure_span(
        &self,
        def_id: DefId,
        target_place: &Place<'tcx>,
        places: &Vec<Operand<'tcx>>,
    ) -> Option<(Span, Span)> {
        debug!(
            "closure_span: def_id={:?} target_place={:?} places={:?}",
            def_id, target_place, places
        );
        let hir_id = self.infcx.tcx.hir().as_local_hir_id(def_id)?;
        let expr = &self.infcx.tcx.hir().expect_expr_by_hir_id(hir_id).node;
        debug!("closure_span: hir_id={:?} expr={:?}", hir_id, expr);
        if let hir::ExprKind::Closure(
            .., args_span, _
        ) = expr {
            let var_span = self.infcx.tcx.with_freevars(
                hir_id,
                |freevars| {
                    for (v, place) in freevars.iter().zip(places) {
                        match place {
                            Operand::Copy(place) |
                            Operand::Move(place) if target_place == place => {
                                debug!("closure_span: found captured local {:?}", place);
                                return Some(v.span);
                            },
                            _ => {}
                        }
                    }

                    None
                },
            )?;

            Some((*args_span, var_span))
        } else {
            None
        }
    }

    /// Helper to retrieve span(s) of given borrow from the current MIR
    /// representation
    pub(super) fn retrieve_borrow_spans(&self, borrow: &BorrowData<'_>) -> UseSpans {
        let span = self.mir.source_info(borrow.reserve_location).span;
        self.borrow_spans(span, borrow.reserve_location)
    }
}
