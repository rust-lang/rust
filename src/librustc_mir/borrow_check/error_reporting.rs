// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::WriteKind;
use rustc::middle::region::ScopeTree;
use rustc::mir::{BorrowKind, Field, Local, LocalKind, Location, Operand};
use rustc::mir::{Place, ProjectionElem, Rvalue, Statement, StatementKind};
use rustc::ty::{self, RegionKind};
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::sync::Lrc;
use syntax_pos::Span;

use super::borrow_set::BorrowData;
use super::{Context, MirBorrowckCtxt};
use super::{InitializationRequiringAction, PrefixSet};

use dataflow::move_paths::MovePathIndex;
use dataflow::{FlowAtLocation, MovingOutStatements};
use util::borrowck_errors::{BorrowckErrors, Origin};

impl<'cx, 'gcx, 'tcx> MirBorrowckCtxt<'cx, 'gcx, 'tcx> {
    pub(super) fn report_use_of_moved_or_uninitialized(
        &mut self,
        _context: Context,
        desired_action: InitializationRequiringAction,
        (place, span): (&Place<'tcx>, Span),
        mpi: MovePathIndex,
        curr_move_out: &FlowAtLocation<MovingOutStatements<'_, 'gcx, 'tcx>>,
    ) {
        let mois = self.move_data.path_map[mpi]
            .iter()
            .filter(|moi| curr_move_out.contains(moi))
            .collect::<Vec<_>>();

        if mois.is_empty() {
            let root_place = self.prefixes(&place, PrefixSet::All).last().unwrap();

            if self.moved_error_reported.contains(&root_place.clone()) {
                debug!(
                    "report_use_of_moved_or_uninitialized place: error about {:?} suppressed",
                    root_place
                );
                return;
            }

            self.moved_error_reported.insert(root_place.clone());

            let item_msg = match self.describe_place_with_options(place, IncludingDowncast(true)) {
                Some(name) => format!("`{}`", name),
                None => "value".to_owned(),
            };
            self.tcx
                .cannot_act_on_uninitialized_variable(
                    span,
                    desired_action.as_noun(),
                    &self
                        .describe_place_with_options(place, IncludingDowncast(true))
                        .unwrap_or("_".to_owned()),
                    Origin::Mir,
                )
                .span_label(span, format!("use of possibly uninitialized {}", item_msg))
                .emit();
        } else {
            let msg = ""; //FIXME: add "partially " or "collaterally "

            let mut err = self.tcx.cannot_act_on_moved_value(
                span,
                desired_action.as_noun(),
                msg,
                self.describe_place_with_options(&place, IncludingDowncast(true)),
                Origin::Mir,
            );

            let mut is_loop_move = false;
            for moi in &mois {
                let move_msg = ""; //FIXME: add " (into closure)"
                let move_span = self
                    .mir
                    .source_info(self.move_data.moves[**moi].source)
                    .span;
                if span == move_span {
                    err.span_label(
                        span,
                        format!("value moved{} here in previous iteration of loop", move_msg),
                    );
                    is_loop_move = true;
                } else {
                    err.span_label(move_span, format!("value moved{} here", move_msg));
                };
            }
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
                    ty::TypeVariants::TyClosure(id, _) => {
                        let tables = self.tcx.typeck_tables_of(id);
                        let node_id = self.tcx.hir.as_local_node_id(id).unwrap();
                        let hir_id = self.tcx.hir.node_to_hir_id(node_id);
                        if let Some(_) = tables.closure_kind_origins().get(hir_id) {
                            false
                        } else {
                            true
                        }
                    }
                    _ => true,
                };

                if needs_note {
                    let mpi = self.move_data.moves[*mois[0]].path;
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

            err.emit();
        }
    }

    pub(super) fn report_move_out_while_borrowed(
        &mut self,
        context: Context,
        (place, span): (&Place<'tcx>, Span),
        borrow: &BorrowData<'tcx>,
    ) {
        let tcx = self.tcx;
        let value_msg = match self.describe_place(place) {
            Some(name) => format!("`{}`", name),
            None => "value".to_owned(),
        };
        let borrow_msg = match self.describe_place(&borrow.borrowed_place) {
            Some(name) => format!("`{}`", name),
            None => "value".to_owned(),
        };
        let mut err = tcx.cannot_move_when_borrowed(
            span,
            &self.describe_place(place).unwrap_or("_".to_owned()),
            Origin::Mir,
        );
        err.span_label(
            self.retrieve_borrow_span(borrow),
            format!("borrow of {} occurs here", borrow_msg),
        );
        err.span_label(span, format!("move out of {} occurs here", value_msg));
        self.explain_why_borrow_contains_point(context, borrow, None, &mut err);
        err.emit();
    }

    pub(super) fn report_use_while_mutably_borrowed(
        &mut self,
        context: Context,
        (place, span): (&Place<'tcx>, Span),
        borrow: &BorrowData<'tcx>,
    ) {
        let tcx = self.tcx;
        let mut err = tcx.cannot_use_when_mutably_borrowed(
            span,
            &self.describe_place(place).unwrap_or("_".to_owned()),
            self.retrieve_borrow_span(borrow),
            &self
                .describe_place(&borrow.borrowed_place)
                .unwrap_or("_".to_owned()),
            Origin::Mir,
        );

        self.explain_why_borrow_contains_point(context, borrow, None, &mut err);

        err.emit();
    }

    /// Finds the span of arguments of a closure (within `maybe_closure_span`) and its usage of
    /// the local assigned at `location`.
    /// This is done by searching in statements succeeding `location`
    /// and originating from `maybe_closure_span`.
    fn find_closure_span(
        &self,
        maybe_closure_span: Span,
        location: Location,
    ) -> Option<(Span, Span)> {
        use rustc::hir::ExprClosure;
        use rustc::mir::AggregateKind;

        let local = match self.mir[location.block]
            .statements
            .get(location.statement_index)
        {
            Some(&Statement {
                kind: StatementKind::Assign(Place::Local(local), _),
                ..
            }) => local,
            _ => return None,
        };

        for stmt in &self.mir[location.block].statements[location.statement_index + 1..] {
            if maybe_closure_span != stmt.source_info.span {
                break;
            }

            if let StatementKind::Assign(_, Rvalue::Aggregate(ref kind, ref places)) = stmt.kind {
                if let AggregateKind::Closure(def_id, _) = **kind {
                    debug!("find_closure_span: found closure {:?}", places);

                    return if let Some(node_id) = self.tcx.hir.as_local_node_id(def_id) {
                        let args_span = if let ExprClosure(_, _, _, span, _) =
                            self.tcx.hir.expect_expr(node_id).node
                        {
                            span
                        } else {
                            return None;
                        };

                        self.tcx
                            .with_freevars(node_id, |freevars| {
                                for (v, place) in freevars.iter().zip(places) {
                                    match *place {
                                        Operand::Copy(Place::Local(l))
                                        | Operand::Move(Place::Local(l)) if local == l =>
                                        {
                                            debug!(
                                                "find_closure_span: found captured local {:?}",
                                                l
                                            );
                                            return Some(v.span);
                                        }
                                        _ => {}
                                    }
                                }
                                None
                            })
                            .map(|var_span| (args_span, var_span))
                    } else {
                        None
                    };
                }
            }
        }

        None
    }

    pub(super) fn report_conflicting_borrow(
        &mut self,
        context: Context,
        (place, span): (&Place<'tcx>, Span),
        gen_borrow_kind: BorrowKind,
        issued_borrow: &BorrowData<'tcx>,
    ) {
        let issued_span = self.retrieve_borrow_span(issued_borrow);

        let new_closure_span = self.find_closure_span(span, context.loc);
        let span = new_closure_span.map(|(args, _)| args).unwrap_or(span);
        let old_closure_span = self.find_closure_span(issued_span, issued_borrow.reserve_location);
        let issued_span = old_closure_span
            .map(|(args, _)| args)
            .unwrap_or(issued_span);

        let desc_place = self.describe_place(place).unwrap_or("_".to_owned());
        let tcx = self.tcx;

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

        if let Some((_, var_span)) = old_closure_span {
            let place = &issued_borrow.borrowed_place;
            let desc_place = self.describe_place(place).unwrap_or("_".to_owned());

            err.span_label(
                var_span,
                format!(
                    "previous borrow occurs due to use of `{}` in closure",
                    desc_place
                ),
            );
        }

        if let Some((_, var_span)) = new_closure_span {
            err.span_label(
                var_span,
                format!("borrow occurs due to use of `{}` in closure", desc_place),
            );
        }

        self.explain_why_borrow_contains_point(context, issued_borrow, None, &mut err);

        err.emit();
    }

    pub(super) fn report_borrowed_value_does_not_live_long_enough(
        &mut self,
        context: Context,
        borrow: &BorrowData<'tcx>,
        place_span: (&Place<'tcx>, Span),
        kind: Option<WriteKind>,
    ) {
        let drop_span = place_span.1;
        let scope_tree = self.tcx.region_scope_tree(self.mir_def_id);
        let root_place = self
            .prefixes(&borrow.borrowed_place, PrefixSet::All)
            .last()
            .unwrap();

        let borrow_span = self.mir.source_info(borrow.reserve_location).span;
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

        match (borrow.region, &self.describe_place(&borrow.borrowed_place)) {
            (RegionKind::ReScope(_), Some(name)) => {
                self.report_scoped_local_value_does_not_live_long_enough(
                    context,
                    name,
                    &scope_tree,
                    &borrow,
                    drop_span,
                    borrow_span,
                    proper_span,
                );
            }
            (RegionKind::ReScope(_), None) => {
                self.report_scoped_temporary_value_does_not_live_long_enough(
                    context,
                    &scope_tree,
                    &borrow,
                    drop_span,
                    borrow_span,
                    proper_span,
                );
            }
            (RegionKind::ReEarlyBound(_), Some(name))
            | (RegionKind::ReFree(_), Some(name))
            | (RegionKind::ReStatic, Some(name))
            | (RegionKind::ReEmpty, Some(name))
            | (RegionKind::ReVar(_), Some(name)) => {
                self.report_unscoped_local_value_does_not_live_long_enough(
                    context,
                    name,
                    &scope_tree,
                    &borrow,
                    drop_span,
                    borrow_span,
                    proper_span,
                    kind.map(|k| (k, place_span.0)),
                );
            }
            (RegionKind::ReEarlyBound(_), None)
            | (RegionKind::ReFree(_), None)
            | (RegionKind::ReStatic, None)
            | (RegionKind::ReEmpty, None)
            | (RegionKind::ReVar(_), None) => {
                self.report_unscoped_temporary_value_does_not_live_long_enough(
                    context,
                    &scope_tree,
                    &borrow,
                    drop_span,
                    borrow_span,
                    proper_span,
                );
            }
            (RegionKind::ReLateBound(_, _), _)
            | (RegionKind::ReSkolemized(_, _), _)
            | (RegionKind::ReClosureBound(_), _)
            | (RegionKind::ReCanonical(_), _)
            | (RegionKind::ReErased, _) => {
                span_bug!(
                    drop_span,
                    "region {:?} does not make sense in this context",
                    borrow.region
                );
            }
        }
    }

    fn report_scoped_local_value_does_not_live_long_enough(
        &mut self,
        context: Context,
        name: &String,
        _scope_tree: &Lrc<ScopeTree>,
        borrow: &BorrowData<'tcx>,
        drop_span: Span,
        borrow_span: Span,
        _proper_span: Span,
    ) {
        let tcx = self.tcx;
        let mut err =
            tcx.path_does_not_live_long_enough(borrow_span, &format!("`{}`", name), Origin::Mir);
        err.span_label(borrow_span, "borrowed value does not live long enough");
        err.span_label(
            drop_span,
            format!("`{}` dropped here while still borrowed", name),
        );
        self.explain_why_borrow_contains_point(context, borrow, None, &mut err);
        err.emit();
    }

    fn report_scoped_temporary_value_does_not_live_long_enough(
        &mut self,
        context: Context,
        _scope_tree: &Lrc<ScopeTree>,
        borrow: &BorrowData<'tcx>,
        drop_span: Span,
        _borrow_span: Span,
        proper_span: Span,
    ) {
        let tcx = self.tcx;
        let mut err =
            tcx.path_does_not_live_long_enough(proper_span, "borrowed value", Origin::Mir);
        err.span_label(proper_span, "temporary value does not live long enough");
        err.span_label(
            drop_span,
            "temporary value dropped here while still borrowed",
        );
        err.note("consider using a `let` binding to increase its lifetime");
        self.explain_why_borrow_contains_point(context, borrow, None, &mut err);
        err.emit();
    }

    fn report_unscoped_local_value_does_not_live_long_enough(
        &mut self,
        context: Context,
        name: &String,
        scope_tree: &Lrc<ScopeTree>,
        borrow: &BorrowData<'tcx>,
        drop_span: Span,
        borrow_span: Span,
        _proper_span: Span,
        kind_place: Option<(WriteKind, &Place<'tcx>)>,
    ) {
        debug!(
            "report_unscoped_local_value_does_not_live_long_enough(\
             {:?}, {:?}, {:?}, {:?}, {:?}, {:?}\
             )",
            context, name, scope_tree, borrow, drop_span, borrow_span
        );

        let tcx = self.tcx;
        let mut err =
            tcx.path_does_not_live_long_enough(borrow_span, &format!("`{}`", name), Origin::Mir);
        err.span_label(borrow_span, "borrowed value does not live long enough");
        err.span_label(drop_span, "borrowed value only lives until here");

        self.explain_why_borrow_contains_point(context, borrow, kind_place, &mut err);
        err.emit();
    }

    fn report_unscoped_temporary_value_does_not_live_long_enough(
        &mut self,
        context: Context,
        scope_tree: &Lrc<ScopeTree>,
        borrow: &BorrowData<'tcx>,
        drop_span: Span,
        _borrow_span: Span,
        proper_span: Span,
    ) {
        debug!(
            "report_unscoped_temporary_value_does_not_live_long_enough(\
             {:?}, {:?}, {:?}, {:?}, {:?}\
             )",
            context, scope_tree, borrow, drop_span, proper_span
        );

        let tcx = self.tcx;
        let mut err =
            tcx.path_does_not_live_long_enough(proper_span, "borrowed value", Origin::Mir);
        err.span_label(proper_span, "temporary value does not live long enough");
        err.span_label(drop_span, "temporary value only lives until here");

        self.explain_why_borrow_contains_point(context, borrow, None, &mut err);
        err.emit();
    }

    pub(super) fn report_illegal_mutation_of_borrowed(
        &mut self,
        context: Context,
        (place, span): (&Place<'tcx>, Span),
        loan: &BorrowData<'tcx>,
    ) {
        let tcx = self.tcx;
        let mut err = tcx.cannot_assign_to_borrowed(
            span,
            self.retrieve_borrow_span(loan),
            &self.describe_place(place).unwrap_or("_".to_owned()),
            Origin::Mir,
        );

        self.explain_why_borrow_contains_point(context, loan, None, &mut err);

        err.emit();
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
        let is_arg = if let Place::Local(local) = place {
            if let LocalKind::Arg = self.mir.local_kind(*local) {
                true
            } else {
                false
            }
        } else {
            false
        };

        let mut err = self.tcx.cannot_reassign_immutable(
            span,
            &self.describe_place(place).unwrap_or("_".to_owned()),
            is_arg,
            Origin::Mir,
        );
        let msg = if is_arg {
            "cannot assign to immutable argument"
        } else {
            "cannot assign twice to immutable variable"
        };
        if span != assigned_span {
            if !is_arg {
                let value_msg = match self.describe_place(place) {
                    Some(name) => format!("`{}`", name),
                    None => "value".to_owned(),
                };
                err.span_label(assigned_span, format!("first assignment to {}", value_msg));
            }
        }
        if let Place::Local(local) = err_place {
            let local_decl = &self.mir.local_decls[*local];
            if let Some(name) = local_decl.name {
                if local_decl.can_be_made_mutable() {
                    err.span_label(
                        local_decl.source_info.span,
                        format!("consider changing this to `mut {}`", name),
                    );
                }
            }
        }
        err.span_label(span, msg);
        err.emit();
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
            Place::Local(local) => {
                self.append_local_to_string(local, buf)?;
            }
            Place::Static(ref static_) => {
                buf.push_str(&format!("{}", &self.tcx.item_name(static_.def_id)));
            }
            Place::Projection(ref proj) => {
                match proj.elem {
                    ProjectionElem::Deref => {
                        if let Some(field) = self.is_upvar_field_projection(&proj.base) {
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

                        if let Some(field) = self.is_upvar_field_projection(place) {
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
                        if let Err(_) = self.append_local_to_string(index, buf) {
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
                buf.push_str(&format!("{}", name));
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
                    format!("{}", self.describe_field(&proj.base, field))
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
                ty::TyAdt(def, _) => if def.is_enum() {
                    format!("{}", field.index())
                } else {
                    format!("{}", def.non_enum_variant().fields[field.index()].ident)
                },
                ty::TyTuple(_) => format!("{}", field.index()),
                ty::TyRef(_, ty, _) | ty::TyRawPtr(ty::TypeAndMut { ty, .. }) => {
                    self.describe_field_from_ty(&ty, field)
                }
                ty::TyArray(ty, _) | ty::TySlice(ty) => self.describe_field_from_ty(&ty, field),
                ty::TyClosure(def_id, _) | ty::TyGenerator(def_id, _, _) => {
                    // Convert the def-id into a node-id. node-ids are only valid for
                    // the local code in the current crate, so this returns an `Option` in case
                    // the closure comes from another crate. But in that case we wouldn't
                    // be borrowck'ing it, so we can just unwrap:
                    let node_id = self.tcx.hir.as_local_node_id(def_id).unwrap();
                    let freevar = self.tcx.with_freevars(node_id, |fv| fv[field.index()]);

                    self.tcx.hir.name(freevar.var_id()).to_string()
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

    // Retrieve span of given borrow from the current MIR representation
    crate fn retrieve_borrow_span(&self, borrow: &BorrowData) -> Span {
        self.mir.source_info(borrow.reserve_location).span
    }

    // Retrieve type of a place for the current MIR representation
    fn retrieve_type_for_place(&self, place: &Place<'tcx>) -> Option<ty::Ty> {
        match place {
            Place::Local(local) => {
                let local = &self.mir.local_decls[*local];
                Some(local.ty)
            }
            Place::Static(ref st) => Some(st.ty),
            Place::Projection(ref proj) => match proj.elem {
                ProjectionElem::Field(_, ty) => Some(ty),
                _ => None,
            },
        }
    }
}
