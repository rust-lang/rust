use rustc_errors::{
    struct_span_err, DiagnosticBuilder, DiagnosticId, DiagnosticMessage, ErrorGuaranteed, MultiSpan,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;

impl<'cx, 'tcx> crate::MirBorrowckCtxt<'cx, 'tcx> {
    pub(crate) fn cannot_move_when_borrowed(
        &self,
        span: Span,
        borrow_span: Span,
        place: &str,
        borrow_place: &str,
        value_place: &str,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        self.infcx.tcx.sess.create_err(crate::session_diagnostics::MoveBorrow {
            place,
            span,
            borrow_place,
            value_place,
            borrow_span,
        })
    }

    pub(crate) fn cannot_use_when_mutably_borrowed(
        &self,
        span: Span,
        desc: &str,
        borrow_span: Span,
        borrow_desc: &str,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let mut err = struct_span_err!(
            self,
            span,
            E0503,
            "cannot use {} because it was mutably borrowed",
            desc,
        );

        err.span_label(borrow_span, format!("{} is borrowed here", borrow_desc));
        err.span_label(span, format!("use of borrowed {}", borrow_desc));
        err
    }

    pub(crate) fn cannot_mutably_borrow_multiply(
        &self,
        new_loan_span: Span,
        desc: &str,
        opt_via: &str,
        old_loan_span: Span,
        old_opt_via: &str,
        old_load_end_span: Option<Span>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let via =
            |msg: &str| if msg.is_empty() { "".to_string() } else { format!(" (via {})", msg) };
        let mut err = struct_span_err!(
            self,
            new_loan_span,
            E0499,
            "cannot borrow {}{} as mutable more than once at a time",
            desc,
            via(opt_via),
        );
        if old_loan_span == new_loan_span {
            // Both borrows are happening in the same place
            // Meaning the borrow is occurring in a loop
            err.span_label(
                new_loan_span,
                format!(
                    "{}{} was mutably borrowed here in the previous iteration of the loop{}",
                    desc,
                    via(opt_via),
                    opt_via,
                ),
            );
            if let Some(old_load_end_span) = old_load_end_span {
                err.span_label(old_load_end_span, "mutable borrow ends here");
            }
        } else {
            err.span_label(
                old_loan_span,
                format!("first mutable borrow occurs here{}", via(old_opt_via)),
            );
            err.span_label(
                new_loan_span,
                format!("second mutable borrow occurs here{}", via(opt_via)),
            );
            if let Some(old_load_end_span) = old_load_end_span {
                err.span_label(old_load_end_span, "first borrow ends here");
            }
        }
        err
    }

    pub(crate) fn cannot_uniquely_borrow_by_two_closures(
        &self,
        new_loan_span: Span,
        desc: &str,
        old_loan_span: Span,
        old_load_end_span: Option<Span>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let mut err = struct_span_err!(
            self,
            new_loan_span,
            E0524,
            "two closures require unique access to {} at the same time",
            desc,
        );
        if old_loan_span == new_loan_span {
            err.span_label(
                old_loan_span,
                "closures are constructed here in different iterations of loop",
            );
        } else {
            err.span_label(old_loan_span, "first closure is constructed here");
            err.span_label(new_loan_span, "second closure is constructed here");
        }
        if let Some(old_load_end_span) = old_load_end_span {
            err.span_label(old_load_end_span, "borrow from first closure ends here");
        }
        err
    }

    pub(crate) fn cannot_uniquely_borrow_by_one_closure(
        &self,
        new_loan_span: Span,
        container_name: &str,
        desc_new: &str,
        opt_via: &str,
        old_loan_span: Span,
        noun_old: &str,
        old_opt_via: &str,
        previous_end_span: Option<Span>,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        let mut err = struct_span_err!(
            self,
            new_loan_span,
            E0500,
            "closure requires unique access to {} but {} is already borrowed{}",
            desc_new,
            noun_old,
            old_opt_via,
        );
        err.span_label(
            new_loan_span,
            format!("{} construction occurs here{}", container_name, opt_via),
        );
        err.span_label(old_loan_span, format!("borrow occurs here{}", old_opt_via));
        if let Some(previous_end_span) = previous_end_span {
            err.span_label(previous_end_span, "borrow ends here");
        }
        err
    }

    pub(crate) fn cannot_reborrow_already_uniquely_borrowed(
        &self,
        new_loan_span: Span,
        container_name: &str,
        desc_new: &str,
        opt_via: &str,
        kind_new: &str,
        old_loan_span: Span,
        old_opt_via: &str,
        previous_end_span: Option<Span>,
        second_borrow_desc: &str,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        let mut err = struct_span_err!(
            self,
            new_loan_span,
            E0501,
            "cannot borrow {}{} as {} because previous closure requires unique access",
            desc_new,
            opt_via,
            kind_new,
        );
        err.span_label(
            new_loan_span,
            format!("{}borrow occurs here{}", second_borrow_desc, opt_via),
        );
        err.span_label(
            old_loan_span,
            format!("{} construction occurs here{}", container_name, old_opt_via),
        );
        if let Some(previous_end_span) = previous_end_span {
            err.span_label(previous_end_span, "borrow from closure ends here");
        }
        err
    }

    pub(crate) fn cannot_reborrow_already_borrowed(
        &self,
        span: Span,
        desc_new: &str,
        msg_new: &str,
        kind_new: &str,
        old_span: Span,
        noun_old: &str,
        kind_old: &str,
        msg_old: &str,
        old_load_end_span: Option<Span>,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        let via =
            |msg: &str| if msg.is_empty() { "".to_string() } else { format!(" (via {})", msg) };
        let mut err = struct_span_err!(
            self,
            span,
            E0502,
            "cannot borrow {}{} as {} because {} is also borrowed as {}{}",
            desc_new,
            via(msg_new),
            kind_new,
            noun_old,
            kind_old,
            via(msg_old),
        );

        if msg_new == "" {
            // If `msg_new` is empty, then this isn't a borrow of a union field.
            err.span_label(span, format!("{} borrow occurs here", kind_new));
            err.span_label(old_span, format!("{} borrow occurs here", kind_old));
        } else {
            // If `msg_new` isn't empty, then this a borrow of a union field.
            err.span_label(
                span,
                format!(
                    "{} borrow of {} -- which overlaps with {} -- occurs here",
                    kind_new, msg_new, msg_old,
                ),
            );
            err.span_label(old_span, format!("{} borrow occurs here{}", kind_old, via(msg_old)));
        }

        if let Some(old_load_end_span) = old_load_end_span {
            err.span_label(old_load_end_span, format!("{} borrow ends here", kind_old));
        }
        err
    }

    pub(crate) fn cannot_assign_to_borrowed(
        &self,
        span: Span,
        borrow_span: Span,
        desc: &str,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        let mut err = struct_span_err!(
            self,
            span,
            E0506,
            "cannot assign to {} because it is borrowed",
            desc,
        );

        err.span_label(borrow_span, format!("{} is borrowed here", desc));
        err.span_label(span, format!("{} is assigned to here but it was already borrowed", desc));
        err
    }

    pub(crate) fn cannot_reassign_immutable(
        &self,
        span: Span,
        desc: &str,
        is_arg: bool,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        let msg = if is_arg { "to immutable argument" } else { "twice to immutable variable" };
        struct_span_err!(self, span, E0384, "cannot assign {} {}", msg, desc)
    }

    pub(crate) fn cannot_assign(
        &self,
        span: Span,
        desc: &str,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        struct_span_err!(self, span, E0594, "cannot assign to {}", desc)
    }

    pub(crate) fn cannot_move_out_of(
        &self,
        move_from_span: Span,
        move_from_desc: &str,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        struct_span_err!(self, move_from_span, E0507, "cannot move out of {}", move_from_desc)
    }

    /// Signal an error due to an attempt to move out of the interior
    /// of an array or slice. `is_index` is None when error origin
    /// didn't capture whether there was an indexing operation or not.
    pub(crate) fn cannot_move_out_of_interior_noncopy(
        &self,
        move_from_span: Span,
        ty: Ty<'_>,
        is_index: Option<bool>,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        let type_name = match (&ty.kind(), is_index) {
            (&ty::Array(_, _), Some(true)) | (&ty::Array(_, _), None) => "array",
            (&ty::Slice(_), _) => "slice",
            _ => span_bug!(move_from_span, "this path should not cause illegal move"),
        };
        let mut err = struct_span_err!(
            self,
            move_from_span,
            E0508,
            "cannot move out of type `{}`, a non-copy {}",
            ty,
            type_name,
        );
        err.span_label(move_from_span, "cannot move out of here");
        err
    }

    pub(crate) fn cannot_move_out_of_interior_of_drop(
        &self,
        move_from_span: Span,
        container_ty: Ty<'_>,
    ) -> DiagnosticBuilder<'cx, ErrorGuaranteed> {
        let mut err = struct_span_err!(
            self,
            move_from_span,
            E0509,
            "cannot move out of type `{}`, which implements the `Drop` trait",
            container_ty,
        );
        err.span_label(move_from_span, "cannot move out of here");
        err
    }

    pub(crate) fn cannot_act_on_moved_value(
        &self,
        use_span: Span,
        verb: &str,
        optional_adverb_for_moved: &str,
        moved_path: Option<String>,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let moved_path = moved_path.map(|mp| format!(": `{}`", mp)).unwrap_or_default();

        struct_span_err!(
            self,
            use_span,
            E0382,
            "{} of {}moved value{}",
            verb,
            optional_adverb_for_moved,
            moved_path,
        )
    }

    pub(crate) fn cannot_borrow_path_as_mutable_because(
        &self,
        span: Span,
        path: &str,
        reason: &str,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        struct_span_err!(self, span, E0596, "cannot borrow {} as mutable{}", path, reason,)
    }

    pub(crate) fn cannot_mutate_in_immutable_section(
        &self,
        mutate_span: Span,
        immutable_span: Span,
        immutable_place: &str,
        immutable_section: &str,
        action: &str,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let mut err = struct_span_err!(
            self,
            mutate_span,
            E0510,
            "cannot {} {} in {}",
            action,
            immutable_place,
            immutable_section,
        );
        err.span_label(mutate_span, format!("cannot {}", action));
        err.span_label(immutable_span, format!("value is immutable in {}", immutable_section));
        err
    }

    pub(crate) fn cannot_borrow_across_generator_yield(
        &self,
        span: Span,
        yield_span: Span,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let mut err = struct_span_err!(
            self,
            span,
            E0626,
            "borrow may still be in use when generator yields",
        );
        err.span_label(yield_span, "possible yield occurs here");
        err
    }

    pub(crate) fn cannot_borrow_across_destructor(
        &self,
        borrow_span: Span,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        struct_span_err!(
            self,
            borrow_span,
            E0713,
            "borrow may still be in use when destructor runs",
        )
    }

    pub(crate) fn path_does_not_live_long_enough(
        &self,
        span: Span,
        path: &str,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        struct_span_err!(self, span, E0597, "{} does not live long enough", path,)
    }

    pub(crate) fn cannot_return_reference_to_local(
        &self,
        span: Span,
        return_kind: &str,
        reference_desc: &str,
        path_desc: &str,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let mut err = struct_span_err!(
            self,
            span,
            E0515,
            "cannot {RETURN} {REFERENCE} {LOCAL}",
            RETURN = return_kind,
            REFERENCE = reference_desc,
            LOCAL = path_desc,
        );

        err.span_label(
            span,
            format!("{}s a {} data owned by the current function", return_kind, reference_desc),
        );

        err
    }

    pub(crate) fn cannot_capture_in_long_lived_closure(
        &self,
        closure_span: Span,
        closure_kind: &str,
        borrowed_path: &str,
        capture_span: Span,
        scope: &str,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let mut err = struct_span_err!(
            self,
            closure_span,
            E0373,
            "{closure_kind} may outlive the current {scope}, but it borrows {borrowed_path}, \
             which is owned by the current {scope}",
        );
        err.span_label(capture_span, format!("{} is borrowed here", borrowed_path))
            .span_label(closure_span, format!("may outlive borrowed value {}", borrowed_path));
        err
    }

    pub(crate) fn thread_local_value_does_not_live_long_enough(
        &self,
        span: Span,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        struct_span_err!(self, span, E0712, "thread-local variable borrowed past end of function",)
    }

    pub(crate) fn temporary_value_borrowed_for_too_long(
        &self,
        span: Span,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        struct_span_err!(self, span, E0716, "temporary value dropped while borrowed",)
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub(crate) fn struct_span_err_with_code<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        self.infcx.tcx.sess.struct_span_err_with_code(sp, msg, code)
    }
}

pub(crate) fn borrowed_data_escapes_closure<'tcx>(
    tcx: TyCtxt<'tcx>,
    escape_span: Span,
    escapes_from: &str,
) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
    struct_span_err!(
        tcx.sess,
        escape_span,
        E0521,
        "borrowed data escapes outside of {}",
        escapes_from,
    )
}
